# AI-Driven Error Correction for Real-Time Wireless Image Transmission  
_Deep convolutional autoencoder with Banach-space contraction principles_

---

## 1. Project Overview
This repository accompanies the paper **“AI-Driven Error Correction for Real-Time Wireless Image Transmission: Integrating Banach Space Principles into Deep Learning.”**

* End-to-end joint source–channel coding for noisy wireless links.  
* Contraction mappings (spectral normalization + dynamic λ) guarantee stable reconstructions.  
* Trained and benchmarked on **CIFAR-10** with Gaussian & burst noise at up to **600 FPS** on an NVIDIA A30.  
* All code, logs, trained weights, and result figures are included for _full reproducibility_.

---

## 2. Repository Layout
├── scripts/ # All runnable Python pipelines
│ ├── autoencoder_simulation.py
│ ├── banach_integration_experiments.py
│ ├── evaluate_test_set.py
│ └── … # more helpers & ablations
├── results/ # Metrics, plots, and sample reconstructions (Git LFS)
│ ├── figures/
│ └── metrics/
├── .gitattributes # LFS tracking rules
└── .gitignore

> **Large files** under `results/**` and `scripts/data/**` are stored with **Git LFS**.  
> Cloning with LFS pulls tiny pointer files first, then streams the blobs.

---
## 3. Quick Start
### 3.1 Clone with LFS
```bash
git lfs install
git clone https://github.com/cgkinyua/deep_learning_project.git
cd deep_learning_project

Setup
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS
pip install -r requirements.txt

4. python scripts/download_cifar10.py

| Step | Script                                      | Purpose                                        |
| ---- | ------------------------------------------- | ---------------------------------------------- |
| 1    | `scripts/autoencoder_simulation.py`         | Train baseline CAE                             |
| 2    | `scripts/banach_integration_experiments.py` | Train Banach-integrated model                  |
| 3    | `scripts/evaluate_test_set.py`              | Compute PSNR / SSIM on test set                |
| 4    | `scripts/extended_evaluation.py`            | Stress-test under varied noise                 |
| 5    | `scripts/generate_baseline_metrics.py`      | Collate CSVs for plots                         |
| 6    | `scripts/psnr_ssim_comparison.py`           | Regenerate paper figures in `results/figures/` |


 5. Hardware & Runtime

    GPU: NVIDIA A30 (24 GB) + CUDA 12

    Training: ≤ 1 h (100 epochs, batch 64)

    Inference: ≈ 0.002 s per 32×32 image (~600 FPS)

    Scripts auto-detect GPU; they fall back to CPU (slower).

6. Key results
| Model                    |      PSNR ↑ |    SSIM ↑ |   FPS ↑ |
| ------------------------ | ----------: | --------: | ------: |
| Baseline CAE             |     20.8 dB |     0.713 |     160 |
| + Iterative Refinement   |     24.0 dB |     0.811 |     320 |
| **+ Banach Contraction** | **28.8 dB** | **0.873** | **600** |

7. How to Cite:
Gitonga C. K., Kwenga I. M., Musundi S.  
“AI-Driven Error Correction for Real-Time Wireless Image Transmission:  
Integrating Banach Space Principles into Deep Learning,” 2025.  
Code and data: https://github.com/cgkinyua/deep_learning_project

8. Contributing

    Fork → feature branch → PR.

    Format with black and check flake8.

    Keep new blobs ≤ 100 MB or add a new LFS rule.

9. License

    Code: MIT License (LICENSE).

    Model weights & result artifacts: CC-BY 4.0.

10. Contact

Open an issue or email cgkinyua@chuka.ac.ke.



