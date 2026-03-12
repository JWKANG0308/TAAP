# TAAP: Timestep-Adaptive Attribute Preservation for Identity-Preserving Face Generation

I implemented TAAP using a standard U-Net with channel multipliers `(1, 2, 4)`, `2` residual blocks per level, and attention at `32×32`. I trained the model for `50` epochs with a model learning rate of `2e-4` and a timestep weighter learning rate of `1e-4`. For sampling, I used `DDIM` with `50` steps.

# Report:

[Project Report](report/TAAP.pdf)

## dataset:
I used the CelebA dataset for this project.  
Dataset page: [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
## Repository structure

```text
TAAP/
├─ train_taap.py
├─ evaluate_taap.py
├─ requirements.txt
├─ .gitignore          
├─ assets/
│  ├─ unused_indices_for_eval.npy
│  └─ taap_samples.png
├─ outputs/
│  ├─ checkpoints/
│  └─ samples/
└─ report/
   └─ TAAP.pdf
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

I evaluated TAAP on CelebA-64 using a 30,000-image training subset. I compared it against a vanilla DDPM baseline and a fixed-weight identity-loss baseline. TAAP achieved the best performance, with an FID of 203.9 and an LPIPS of 0.383, outperforming vanilla DDPM (FID 257.7, LPIPS 0.428) and the fixed-weight baseline (FID 410.5, LPIPS 0.444).

Example qualitative samples from the project are stored in:

```text
assets/taap_samples.png
```

I found that TAAP generated clearer facial features and more distinguishable identities than both the vanilla DDPM and the fixed-weight identity-loss baseline.

## Files included in this repo

- `train_taap.py`: main training script
- `evaluate_taap.py`: evaluation script for FID and LPIPS
- `assets/unused_indices_for_eval.npy`: evaluation subset indices
- `assets/taap_samples.png`: sample generated outputs
- `report/TAAP_report.pdf`: final report

## Notes

- I converted this code from a Colab notebook into plain Python scripts.
- I replaced Google Drive paths such as `/content/drive/...` with repo-relative paths.
- If you are using GitHub web upload, create the folder structure first, then upload each file into the matching folder.
- I did not upload dataset due to it's size

## Citation

If you use or refer to this project, please cite the report in `report/TAAP_report.pdf`.

