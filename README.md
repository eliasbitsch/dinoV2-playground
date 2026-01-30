# DINOv2 Background Removal

Background removal using DINOv2 feature similarity. Compares patch embeddings from camera images against a reference object to generate foreground masks.

## Quick Start

**GPU (recommended):**
```bash
docker compose --profile gpu up --build
```

**CPU only:**
```bash
docker compose --profile cpu up --build
```

Open Jupyter at http://localhost:8888

## Project Structure

| File | Description |
|------|-------------|
| `benchmark_scientific.ipynb` | Main benchmark notebook (300 configs) |
| `benchmark_bg_vs_nbg.ipynb` | BG vs NBG comparison |
| `ref_car/` | Reference images (with/without background) |
| `urbansyn_subset/` | Test images with ground truth masks |
| `benchmark_results.csv` | Benchmark output data |

## Key Findings

- **Best config:** NBG reference, threshold 0.4, blur σ=0.5 → 71.3% IoU
- **Critical factor:** Background-free reference image (+10pp IoU vs with background)
- **Post-processing:** Minimal impact (<1% improvement)

## Requirements

- Docker with NVIDIA runtime (GPU) or Docker (CPU)
- ~4GB disk for DINOv2 model weights

## How It Works

1. Extract global feature from reference image (DINOv2 CLS token)
2. Extract patch features from input image (37×37 grid)
3. Compute cosine similarity between reference and each patch
4. Threshold similarity map to create binary mask
5. Upscale mask to original resolution
