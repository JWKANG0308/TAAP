import argparse
import math
import os
import random
import shutil
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_fid import fid_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm


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
        return self.mlp(self.embed(t)).squeeze(-1)


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

        self.norm1 = GroupNorm(in_channels)
        self.conv1 = conv2d(in_channels, out_channels)
        self.temb_proj = linear(temb_channels, out_channels)
        self.norm2 = GroupNorm(out_channels)
        self.dropout_layer = nn.Dropout(dropout)
        self.conv2 = conv2d(out_channels, out_channels, init_scale=0.0)
        if in_channels != out_channels:
            self.nin_shortcut = nin(in_channels, out_channels)

    def forward(self, x, temb):
        h = self.norm1(x)
        h = swish(h)
        h = self.conv1(h)
        h = h + self.temb_proj(swish(temb))[:, :, None, None]
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
        self.num_levels = len(ch_mult)
        self.temb_ch = ch * 4

        self.temb_dense0 = linear(ch, self.temb_ch)
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
        down_channels = [ch]
        for i_level in range(self.num_levels):
            out_ch = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                down_channels.append(out_ch)
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
                h = torch.cat([h, skips.pop()], dim=1)
                h = layer(h, temb)
            else:
                h = layer(h)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


def get_beta_alpha_linear(beta_start=1e-4, beta_end=0.02, num_timesteps=1000):
    beta = torch.linspace(beta_start, beta_end, num_timesteps)
    alpha = 1 - beta
    alphas_cumprod = torch.cumprod(alpha, dim=0)
    return beta, alpha, alphas_cumprod


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
    num_timesteps = alphas_cumprod.shape[0]
    x = torch.randn(shape, device=device)
    idx_lin = torch.linspace(0, num_timesteps - 1, steps=ddim_steps + 1, device=device)
    idx0 = idx_lin.round().long()
    idx0 = torch.cat([torch.tensor([0, num_timesteps - 1], device=device, dtype=torch.long), idx0]).unique(sorted=True)

    seq_asc = idx0 + 1
    seq_rev = torch.flip(seq_asc, dims=[0])
    seq = torch.cat([seq_rev, torch.tensor([0], device=device, dtype=torch.long)])

    prev_t = seq[0].item()
    for next_t in tqdm(seq[1:], desc="Sampling", leave=False):
        x = p_sample_ddim(model, x, prev_t, next_t.item(), alphas_cumprod, eta)
        prev_t = next_t.item()

    return x.cpu().detach().clamp(-1, 1)


class SingleImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".png", ".jpg", ".jpeg"))])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.paths[idx]


def load_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TAAP checkpoint using FID and LPIPS.")
    parser.add_argument("--celeba_path", type=str, default="data/img_align_celeba")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--unused_idx_path", type=str, default="assets/unused_indices_for_eval.npy")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--num_eval_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    sampled_folder = output_dir / "real_celebA_sampled"
    generated_folder = output_dir / "generated_images"
    sampled_folder.mkdir(parents=True, exist_ok=True)
    generated_folder.mkdir(parents=True, exist_ok=True)

    _, _, alphas_cumprod = get_beta_alpha_linear()
    alphas_cumprod = alphas_cumprod.to(device)

    model = DDPMModel().to(device)
    model = load_checkpoint(args.checkpoint_path, model, device)

    unused_indices = np.load(args.unused_idx_path)
    real_folder = args.celeba_path

    print(f"Loading {len(unused_indices)} unused indices for evaluation")
    sample_indices = random.sample(list(unused_indices), min(args.num_eval_samples, len(unused_indices)))

    real_imgs = sorted([f for f in os.listdir(real_folder) if f.endswith(".jpg")])
    print(f"Copying {len(sample_indices)} real images for evaluation...")
    for i, idx in enumerate(sample_indices):
        fname = real_imgs[idx]
        src = os.path.join(real_folder, fname)
        dst = sampled_folder / f"real_{i:04d}.jpg"
        shutil.copy(src, dst)

    print(f"Generating {len(sample_indices)} synthetic images...")
    generated_count = 0
    for batch_idx in range(0, len(sample_indices), args.batch_size):
        n = min(args.batch_size, len(sample_indices) - batch_idx)
        fake_images = sample_ddim(model, (n, 3, 64, 64), alphas_cumprod, device, ddim_steps=args.ddim_steps, eta=0.0)
        fake_images = (fake_images + 1) / 2
        for i in range(n):
            img = transforms.ToPILImage()(fake_images[i])
            save_path = generated_folder / f"gen_{generated_count:04d}.jpg"
            img.save(save_path)
            generated_count += 1

    print("Computing LPIPS...")
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    gen_ds = SingleImageDataset(generated_folder, transform=transform)
    real_ds = SingleImageDataset(sampled_folder, transform=transform)
    n_pairs = min(len(gen_ds), len(real_ds))

    gen_loader = DataLoader(Subset(gen_ds, list(range(n_pairs))), batch_size=args.batch_size, shuffle=False)
    real_loader = DataLoader(Subset(real_ds, list(range(n_pairs))), batch_size=args.batch_size, shuffle=False)

    total_lpips = []
    with torch.no_grad():
        for (g_img, _), (r_img, _) in zip(gen_loader, real_loader):
            g_img = g_img.to(device)
            r_img = r_img.to(device)
            dist = lpips_fn(g_img, r_img)
            total_lpips.extend(dist.cpu().numpy())

    lpips_scores = np.array(total_lpips)
    print(f"Evaluated {n_pairs} pairs")
    print(f"Mean LPIPS: {lpips_scores.mean():.4f}")
    print(f"Median LPIPS: {np.median(lpips_scores):.4f}")
    print(f"Std LPIPS: {lpips_scores.std():.4f}")

    print("Computing FID...")
    try:
        fid_value = fid_score.calculate_fid_given_paths(
            [str(sampled_folder), str(generated_folder)],
            batch_size=50,
            device=device,
            dims=2048,
        )
        print(f"FID Score: {fid_value:.4f}")
    except Exception as e:
        fid_value = None
        print(f"Error computing FID: {e}")

    results_file = output_dir / "evaluation_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("TAAP DDPM Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Num Samples: {n_pairs}\n")
        f.write("\nLPIPS Scores:\n")
        f.write(f"  Mean: {lpips_scores.mean():.4f}\n")
        f.write(f"  Median: {np.median(lpips_scores):.4f}\n")
        f.write(f"  Std: {lpips_scores.std():.4f}\n")
        if fid_value is not None:
            f.write(f"\nFID Score: {fid_value:.4f}\n")
        else:
            f.write("\nFID Score: Failed to compute\n")

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
