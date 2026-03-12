import argparse
import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CelebADataset(Dataset):
    def __init__(self, img_dir, img_num_list, transform=None):
        self.img_dir = Path(img_dir)
        self.img_paths = [self.img_dir / f"{i:06d}.jpg" for i in img_num_list]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class TimestepAttributeWeighter(nn.Module):
    def __init__(self, num_timesteps=1000):
        super().__init__()
        self.embed = nn.Embedding(num_timesteps + 1, 32)
        self.mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, t):
        t_emb = self.embed(t)
        alpha = self.mlp(t_emb).squeeze(-1)
        return alpha


def swish(x):
    return x * torch.sigmoid(x)


def get_timestep_embedding(t, channel):
    half = channel // 2
    device = t.device
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, dtype=torch.float32, device=device) * -emb)
    emb = t[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups=8, eps=1e-6):
        super().__init__(num_groups, num_channels, eps=eps)


def conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, init_scale=1.0):
    conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
    with torch.no_grad():
        conv.weight.data *= init_scale
    return conv


def nin(in_ch, out_ch, init_scale=1.0):
    layer = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
    with torch.no_grad():
        layer.weight.data *= init_scale
    return layer


def linear(in_features, out_features, init_scale=1.0):
    fc = nn.Linear(in_features, out_features)
    with torch.no_grad():
        fc.weight.data *= init_scale
    return fc


class DownsampleBlock(nn.Module):
    def __init__(self, channels, with_conv=True):
        super().__init__()
        if with_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=256, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_channels = temb_channels
        self.dropout = dropout

        self.norm1 = GroupNorm(in_channels)
        self.conv1 = conv2d(in_channels, out_channels)
        self.temb_proj = linear(temb_channels, out_channels)
        self.norm2 = GroupNorm(out_channels)
        self.dropout_layer = nn.Dropout(dropout)
        self.conv2 = conv2d(out_channels, out_channels, init_scale=0.0)
        if in_channels != out_channels:
            self.nin_shortcut = nin(in_channels, out_channels)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        temb = self.temb_proj(swish(temb))[:, :, None, None]
        h = h + temb

        h = self.norm2(h)
        h = swish(h)
        h = self.dropout_layer(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = GroupNorm(channels)
        self.q = nin(channels, channels)
        self.k = nin(channels, channels)
        self.v = nin(channels, channels)
        self.proj_out = nin(channels, channels, init_scale=0.0)

    def forward(self, x):
        b, c, h, w = x.shape
        h_ = self.norm(x)
        q = self.q(h_).view(b, c, h * w)
        k = self.k(h_).view(b, c, h * w)
        v = self.v(h_).view(b, c, h * w)

        attn = torch.bmm(q.transpose(1, 2), k) * (c ** (-0.5))
        attn = F.softmax(attn, dim=-1)
        h_ = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


class DDPMModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        ch=64,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions={32},
        dropout=0.0,
        resamp_with_conv=False,
        init_resolution=64,
    ):
        super().__init__()
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.num_levels = len(ch_mult)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resamp_with_conv = resamp_with_conv
        self.init_resolution = init_resolution
        self.temb_ch = ch * 4

        self.temb_dense0 = linear(self.ch, self.temb_ch)
        self.temb_dense1 = linear(self.temb_ch, self.temb_ch)
        self.conv_in = conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        self.down_blocks = nn.ModuleList()
        in_ch = ch
        for i_level in range(self.num_levels):
            out_ch = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResnetBlock(in_ch, out_ch, self.temb_ch, dropout))
                curr_resolution = init_resolution // (2 ** i_level)
                if curr_resolution in attn_resolutions:
                    self.down_blocks.append(AttnBlock(out_ch))
                in_ch = out_ch
            if i_level != self.num_levels - 1:
                self.down_blocks.append(DownsampleBlock(out_ch, resamp_with_conv))

        mid_ch = in_ch
        self.mid_block = nn.ModuleList(
            [
                ResnetBlock(mid_ch, mid_ch, self.temb_ch, dropout),
                AttnBlock(mid_ch),
                ResnetBlock(mid_ch, mid_ch, self.temb_ch, dropout),
            ]
        )

        self.up_blocks = nn.ModuleList()
        down_channels = []
        temp_ch = ch
        down_channels.append(temp_ch)

        for i_level in range(self.num_levels):
            out_ch = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                down_channels.append(out_ch)
                temp_ch = out_ch
            if i_level != self.num_levels - 1:
                down_channels.append(out_ch)

        in_ch = mid_ch
        for i_level in reversed(range(self.num_levels)):
            out_ch = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                skip_ch = down_channels.pop()
                self.up_blocks.append(ResnetBlock(in_ch + skip_ch, out_ch, self.temb_ch, dropout))
                curr_resolution = init_resolution // (2 ** i_level)
                if curr_resolution in attn_resolutions:
                    self.up_blocks.append(AttnBlock(out_ch))
                in_ch = out_ch
            if i_level != 0:
                self.up_blocks.append(UpsampleBlock(out_ch, resamp_with_conv))

        self.norm_out = GroupNorm(ch)
        self.conv_out = conv2d(ch, out_channels, kernel_size=3, stride=1, padding=1, init_scale=0.0)

    def forward(self, x, t):
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb_dense0(temb)
        temb = swish(temb)
        temb = self.temb_dense1(temb)

        skips = []
        h = self.conv_in(x)
        skips.append(h)

        for layer in self.down_blocks:
            if isinstance(layer, ResnetBlock):
                h = layer(h, temb)
                skips.append(h)
            elif isinstance(layer, DownsampleBlock):
                h = layer(h)
                skips.append(h)
            else:
                h = layer(h)

        for layer in self.mid_block:
            if isinstance(layer, ResnetBlock):
                h = layer(h, temb)
            else:
                h = layer(h)

        for layer in self.up_blocks:
            if isinstance(layer, ResnetBlock):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, temb)
            else:
                h = layer(h)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


def get_beta_alpha_linear(beta_start=0.0001, beta_end=0.02, num_timesteps=1000):
    betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)
    betas = torch.tensor(betas)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def q_sample(x0, t, noise, alphas_cumprod):
    alpha_bar = alphas_cumprod[t - 1].to(x0.device)
    alpha_bar = alpha_bar.reshape((alpha_bar.size(0), 1, 1, 1))
    return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise


@torch.no_grad()
def p_sample_ddim(model, x_t, t_cur, t_prev, alphas_cumprod, eta=0.0):
    alpha_bar_t = alphas_cumprod[t_cur - 1]
    alpha_bar_prev = alphas_cumprod[t_prev - 1] if t_prev > 0 else torch.tensor(1.0, device=x_t.device)

    sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
    eps_theta = model(x_t, torch.full((x_t.shape[0],), t_cur, device=x_t.device, dtype=torch.long))

    x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1, 1) * eps_theta) / torch.sqrt(alpha_bar_t).view(-1, 1, 1, 1)
    x0_pred = torch.clamp(x0_pred, -3, 3)

    dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma_t**2, min=0.0)).view(-1, 1, 1, 1) * eps_theta
    noise = torch.randn_like(x_t) if t_prev > 0 else torch.zeros_like(x_t)

    x_prev = torch.sqrt(alpha_bar_prev).view(-1, 1, 1, 1) * x0_pred + dir_xt + sigma_t.view(-1, 1, 1, 1) * noise
    x_prev = torch.clamp(x_prev, -4, 4)
    return x_prev


@torch.no_grad()
def sample_ddim(model, shape, alphas_cumprod, device, ddim_steps=50, eta=0.0):
    model.eval()
    num_timesteps = alphas_cumprod.shape[0]
    x = torch.randn(shape, device=device)

    idx_lin = torch.linspace(0, num_timesteps - 1, steps=ddim_steps + 1, device=device)
    idx0 = idx_lin.round().long()
    idx0 = torch.cat([torch.tensor([0, num_timesteps - 1], device=device, dtype=torch.long), idx0]).unique(sorted=True)

    seq_asc = idx0 + 1
    seq_rev = torch.flip(seq_asc, dims=[0])
    seq = torch.cat([seq_rev, torch.tensor([0], device=device, dtype=torch.long)])

    prev_t = seq[0].item()
    for next_t in seq[1:]:
        x = p_sample_ddim(model, x, prev_t, next_t.item(), alphas_cumprod, eta)
        prev_t = next_t.item()

    return x.cpu().detach().clamp(-1, 1)


def make_identity_encoder(device):
    encoder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def get_identity_embedding(img_tensor, identity_encoder):
    img_resized = F.interpolate(img_tensor, size=(160, 160), mode="bilinear")
    mean = torch.tensor([0.5, 0.5, 0.5], device=img_tensor.device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=img_tensor.device).reshape(1, 3, 1, 1)
    img_norm = (img_resized * std) + mean
    return identity_encoder(img_norm)


def compute_taap_loss(model, x_t, t, eps, alphas_cumprod, image, identity_encoder, weight_module):
    pred_eps = model(x_t, t)
    ddpm_loss = F.mse_loss(pred_eps, eps)

    with torch.no_grad():
        id_real = get_identity_embedding(image, identity_encoder)

    alpha_bar = alphas_cumprod[t - 1].view(-1, 1, 1, 1)
    x0_pred = (x_t - torch.sqrt(1 - alpha_bar) * pred_eps) / torch.sqrt(alpha_bar)
    id_gen = get_identity_embedding(x0_pred, identity_encoder)
    alpha_t = weight_module(t)
    cos_sim = F.cosine_similarity(id_real, id_gen, dim=1)
    id_loss = 1 - cos_sim.mean()
    total_loss = ddpm_loss + (alpha_t.mean() * id_loss)
    return total_loss


def compute_mse_loss(model, x_t, t, eps):
    pred_eps = model(x_t, t)
    return F.mse_loss(pred_eps, eps)


def train_epoch_vanilla(model, train_loader, alphas_cumprod, device, optimizer, use_gradient_clipping=True):
    train_loss_sum = 0.0
    train_loss_cnt = 0
    model.train()

    for image in tqdm(train_loader, desc="Training"):
        image = image.to(device)
        eps = torch.randn(image.shape, device=device)
        t = torch.randint(1, 1001, (image.size(0),), dtype=torch.long, device=device)
        x_t = q_sample(image, t, eps, alphas_cumprod)
        loss = compute_mse_loss(model, x_t, t, eps)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        train_loss_sum += loss.item()
        train_loss_cnt += 1

    return train_loss_sum / train_loss_cnt


def train_epoch_taap(model, train_loader, alphas_cumprod, device, optimizer, weight_module, weight_optimizer, identity_encoder, use_gradient_clipping=True):
    train_loss_sum = 0.0
    train_loss_cnt = 0
    model.train()
    weight_module.train()

    for image in tqdm(train_loader, desc="Training"):
        image = image.to(device)
        eps = torch.randn(image.shape, device=device)
        t = torch.randint(1, 1001, (image.size(0),), dtype=torch.long, device=device)
        x_t = q_sample(image, t, eps, alphas_cumprod)
        loss = compute_taap_loss(model, x_t, t, eps, alphas_cumprod, image, identity_encoder, weight_module)

        optimizer.zero_grad(set_to_none=True)
        weight_optimizer.zero_grad()
        loss.backward()

        if use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(weight_module.parameters(), 1.0)

        optimizer.step()
        weight_optimizer.step()
        train_loss_sum += loss.item()
        train_loss_cnt += 1

    return train_loss_sum / train_loss_cnt


def test_epoch(model, test_loader, alphas_cumprod, device):
    test_loss_sum = 0.0
    test_loss_cnt = 0
    model.eval()

    with torch.no_grad():
        for image in tqdm(test_loader, desc="Evaluating"):
            image = image.to(device)
            eps = torch.randn(image.shape, device=device)
            t = torch.randint(1, 1001, (image.size(0),), dtype=torch.long, device=device)
            x_t = q_sample(image, t, eps, alphas_cumprod)
            loss = compute_mse_loss(model, x_t, t, eps)
            test_loss_sum += loss.item()
            test_loss_cnt += 1

    return test_loss_sum / test_loss_cnt


def build_dataloaders(data_dir, num_data=30000, train_test_ratio=0.9, batch_size=64):
    train_transform = transforms.Compose(
        [
            transforms.Resize([64, 64]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    train_img_num = np.arange(1, num_data + 1)[: int(num_data * train_test_ratio)]
    test_img_num = np.arange(1, num_data + 1)[int(num_data * train_test_ratio) :]

    train_dataset = CelebADataset(data_dir, img_num_list=train_img_num, transform=train_transform)
    test_dataset = CelebADataset(data_dir, img_num_list=test_img_num, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


def train(
    model,
    train_loader,
    test_loader,
    alphas_cumprod,
    device,
    optimizer,
    output_dir,
    num_epochs=50,
    save_model_cycle=5,
    use_gradient_clipping=True,
    mode="taap",
    weight_module=None,
    weight_optimizer=None,
    identity_encoder=None,
):
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    samples_dir = output_dir / "samples"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    train_losses = []
    test_losses = []

    if mode == "taap":
        assert weight_module is not None
        assert weight_optimizer is not None
        assert identity_encoder is not None

    for epoch in range(1, num_epochs + 1):
        if mode == "vanilla":
            train_loss = train_epoch_vanilla(model, train_loader, alphas_cumprod, device, optimizer, use_gradient_clipping)
            mode_name = "Vanilla"
        else:
            train_loss = train_epoch_taap(
                model,
                train_loader,
                alphas_cumprod,
                device,
                optimizer,
                weight_module,
                weight_optimizer,
                identity_encoder,
                use_gradient_clipping,
            )
            mode_name = "TAAP"

        test_loss = test_epoch(model, test_loader, alphas_cumprod, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"[{mode_name}] Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        samples = sample_ddim(model, shape=(8, 3, 64, 64), alphas_cumprod=alphas_cumprod, device=device, ddim_steps=50, eta=0.0)
        grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
        torchvision.utils.save_image(grid, samples_dir / f"samples_{mode_name.lower()}_epoch_{epoch}.png")

        if epoch % save_model_cycle == 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "mode": mode,
            }
            if mode == "taap":
                checkpoint["weight_module_state_dict"] = weight_module.state_dict()
                checkpoint["weight_optimizer_state_dict"] = weight_optimizer.state_dict()

            model_path = checkpoints_dir / f"model_{mode_name}_{epoch}.pth"
            torch.save(checkpoint, model_path)
            print(f"Model saved at {model_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", marker="o", markersize=4)
    plt.plot(test_losses, label="Test Loss", marker="s", markersize=4)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"DDPM {mode_name} - Train/Test Loss Curve", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"loss_curve_{mode_name.lower()}.png", dpi=150)
    plt.close()
    print(f"Loss curve saved to {output_dir / f'loss_curve_{mode_name.lower()}.png'}")

    return train_losses, test_losses


def parse_args():
    parser = argparse.ArgumentParser(description="Train TAAP / vanilla DDPM on CelebA-64.")
    parser.add_argument("--data_dir", type=str, default="data/img_align_celeba", help="Folder with CelebA JPG images.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Folder for checkpoints, samples, and curves.")
    parser.add_argument("--mode", type=str, default="taap", choices=["taap", "vanilla"], help="Training mode.")
    parser.add_argument("--num_data", type=int, default=30000, help="Number of images to use from CelebA.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--save_model_cycle", type=int, default=5)
    parser.add_argument("--lr_model", type=float, default=2e-4)
    parser.add_argument("--lr_weight", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Optional checkpoint to resume from.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        num_data=args.num_data,
        batch_size=args.batch_size,
    )

    _, _, alphas_cumprod = get_beta_alpha_linear()
    alphas_cumprod = alphas_cumprod.to(device)

    model = DDPMModel(
        ch=64,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions={32},
        dropout=0.0,
        resamp_with_conv=False,
        init_resolution=64,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr_model, weight_decay=args.weight_decay)

    weight_module = None
    weight_optimizer = None
    identity_encoder = None

    if args.mode == "taap":
        weight_module = TimestepAttributeWeighter(num_timesteps=1000).to(device)
        weight_optimizer = optim.Adam(weight_module.parameters(), lr=args.lr_weight)
        identity_encoder = make_identity_encoder(device)

    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if args.mode == "taap" and "weight_module_state_dict" in checkpoint:
            weight_module.load_state_dict(checkpoint["weight_module_state_dict"])
        if args.mode == "taap" and "weight_optimizer_state_dict" in checkpoint:
            weight_optimizer.load_state_dict(checkpoint["weight_optimizer_state_dict"])
        print(f"Resumed from {args.resume_checkpoint}")

    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        alphas_cumprod=alphas_cumprod,
        device=device,
        optimizer=optimizer,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        save_model_cycle=args.save_model_cycle,
        use_gradient_clipping=True,
        mode=args.mode,
        weight_module=weight_module,
        weight_optimizer=weight_optimizer,
        identity_encoder=identity_encoder,
    )


if __name__ == "__main__":
    main()
