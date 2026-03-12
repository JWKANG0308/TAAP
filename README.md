
# TAAP: Timestep-Adaptive Attribute Preservation for Identity-Preserving Face Generation

TAAP is a diffusion-based face generation project that improves identity preservation by learning a timestep-specific weight for the identity loss during training.

# Report:

[Project Report](report/TAAP.pdf)


## Project summary

Standard DDPMs can generate realistic faces, but they often fail to preserve semantic attributes such as identity. TAAP adds a lightweight **Timestep Attribute Weighter** that learns how important identity supervision should be at each diffusion timestep.

Instead of using a fixed loss:

`L_total = L_DDPM + λ · L_identity`

TAAP learns a timestep-dependent weight:

`L_total = L_DDPM + α_t · L_identity`

According to the report, TAAP was evaluated on **CelebA-64**, trained on a subset of **30,000 images**, and compared against vanilla DDPM and a fixed-weight identity-loss baseline. The report states that TAAP achieved **FID 203.9** and **LPIPS 0.383**, outperforming vanilla DDPM (**FID 257.7, LPIPS 0.428**) and the fixed-weight baseline (**FID 410.5, LPIPS 0.444**). fileciteturn0file0L84-L101

The report also states that the implementation used:
- a standard U-Net with channel multipliers `(1, 2, 4)`
- `2` residual blocks per level
- attention at `32×32`
- `50` training epochs
- model learning rate `2e-4`
- timestep weighter learning rate `1e-4`
- `DDIM` sampling with `50` steps fileciteturn0file0L156-L175

## Repository structure

```text
TAAP/
├─ train_taap.py
├─ evaluate_taap.py
├─ requirements.txt
├─ .gitignore
├─ data/
│  └─ img_align_celeba/              
├─ assets/
│  ├─ unused_indices_for_eval.npy
│  └─ taap_samples.png
├─ outputs/
│  ├─ checkpoints/
│  └─ samples/
├─ evaluation_results/
└─ report/
   └─ TAAP_report.pdf
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Dataset

Put the CelebA images inside:

```text
data/img_align_celeba/
```

Your image files should look like this:

```text
data/img_align_celeba/000001.jpg
data/img_align_celeba/000002.jpg
data/img_align_celeba/000003.jpg
...
```

## Training

Run TAAP training:

```bash
python train_taap.py --data_dir data/img_align_celeba --output_dir outputs --mode taap
```

Run vanilla DDPM training:

```bash
python train_taap.py --data_dir data/img_align_celeba --output_dir outputs --mode vanilla
```

Useful optional arguments:

```bash
python train_taap.py \
  --data_dir data/img_align_celeba \
  --output_dir outputs \
  --mode taap \
  --num_data 30000 \
  --batch_size 64 \
  --num_epochs 50 \
  --save_model_cycle 5
```

To resume from a checkpoint:

```bash
python train_taap.py \
  --data_dir data/img_align_celeba \
  --output_dir outputs \
  --mode taap \
  --resume_checkpoint outputs/checkpoints/model_TAAP_50.pth
```

## Evaluation

Evaluate a saved checkpoint with LPIPS and FID:

```bash
python evaluate_taap.py \
  --celeba_path data/img_align_celeba \
  --checkpoint_path outputs/checkpoints/model_TAAP_50.pth \
  --unused_idx_path assets/unused_indices_for_eval.npy \
  --output_dir evaluation_results
```

## Results

Example qualitative samples from the project are stored in:

```text
assets/taap_samples.png
```

The report notes that TAAP generated clearer facial features and more distinguishable identities than both the vanilla DDPM and the fixed-weight identity-loss baseline. fileciteturn0file0L102-L116

## Files included in this repo

- `train_taap.py`: main training script
- `evaluate_taap.py`: evaluation script for FID and LPIPS
- `assets/unused_indices_for_eval.npy`: evaluation subset indices
- `assets/taap_samples.png`: sample generated outputs
- `report/TAAP_report.pdf`: final report

## Notes

- This code is adapted from a Colab notebook into plain Python scripts.
- Google Drive paths such as `/content/drive/...` were replaced with repo-relative paths.
- If you are using GitHub web upload, create the folder structure first, then upload each file into the matching folder.
- If CelebA is too large for GitHub, do **not** upload the full dataset to GitHub. Keep it locally and add it to `.gitignore`.

## Citation

If you use or refer to this project, please cite the report in `report/TAAP_report.pdf`.
