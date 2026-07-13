# Fusion of UAV–Sentinel‑2 images using convolutional neural networks forremote-sensing analysis
This repository contains the implementation of the research project "Fusion of UAV-Sentinel-2 images using convolutional neural networks for remote-sensing analysis", published in the International Journal of Image and Data Fusion (2026). The study proposes a feature-level fusion methodology based on vegetation indices, employing super-resolution models (FSRCNN, VDSR, and ESRGAN) to enhance the spatial resolution of Sentinel-2 imagery using high-fidelity UAV data.

Academic Reference
If you use this code in your research work, please cite the original article:

Murguia-Cozar, A., López-Canteñs, G. J., Macedo-Cruz, A., Velazquez-López, N., & Ontiveros-Capurata, R. E. (2026). Fusion of UAV-Sentinel-2 images using convolutional neural networks for remote-sensing analysis. International Journal of Image and Data Fusion, 17(1), 2701715. https://doi.org/10.1080/19479832.2026.2701715

Technical Features
Designed for: Reading multiband Sentinel–UAV pairs (same grid) and training on image patches.

Models: Implementations of FSRCNN, VDSR, and ESRGAN adapted for remote sensing applications.

Model Management: Automatic saving of models with resolution and acquisition dates in the filename.

Evaluation: Includes an inference function that integrates evaluate_metrics.py for post-inference evaluation (RMSE, PSNR, SSIM, Correlation).

Installation
To install the necessary dependencies, run the following command in your terminal:

pip install -r requirements.txt
