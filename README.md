# Fusion of UAV–Sentinel‑2 images using convolutional neural networks forremote-sensing analysis
<h5><p align="justify"> This repository contains the implementation of the research project "Fusion of UAV-Sentinel-2 images using convolutional neural networks for remote-sensing analysis", published in the International Journal of Image and Data Fusion (2026). The study proposes a feature-level fusion methodology based on vegetation indices, employing super-resolution models (FSRCNN, VDSR, and ESRGAN) to enhance the spatial resolution of Sentinel-2 imagery using high-fidelity UAV data. </p>

<h3> 📄 Academic Reference 
<h5> <p align="justify"> If you use this code in your research work, please cite the original article:

Murguia-Cozar, A., López-Canteñs, G. J., Macedo-Cruz, A., Velazquez-López, N., & Ontiveros-Capurata, R. E. (2026). Fusion of UAV-Sentinel-2 images using convolutional neural networks for remote-sensing analysis. International Journal of Image and Data Fusion, 17(1), 2701715. https://doi.org/10.1080/19479832.2026.2701715 </p>

<h3> 🛠️ Technical Features
<h5> <p align="justify"> Designed for: Reading multiband Sentinel–UAV pairs (same grid) and training on image patches.

Models: Implementations of FSRCNN, VDSR, and ESRGAN adapted for remote sensing applications.

Model Management: Automatic saving of models with resolution and acquisition dates in the filename.

Evaluation: Includes an inference function that integrates evaluate_metrics.py for post-inference evaluation (RMSE, PSNR, SSIM, Correlation).</p>

<h3> 🚀 Installation
<h5> <p align="justify"> To install the necessary dependencies, run the following command in your terminal:

pip install -r requirements.txt</p>

<h3>📊 Dataset
<h5> <p align="justify"> Due to the large data volume (50 MB per image), the full dataset is not included in this repository to optimize version control.

Samples: Two images are included (20250803_comp_dron_2m.tif and 20250803_comp_sentinel_2m.tif) with test files to verify code execution.

Full Dataset Access: If you require access to the full dataset used in this research, please send a request to: alvaro.murguia1989@gmail.com </p>

<h3>🤝 Contributions
<h5> <p align="justify"> The authors welcome any collaboration or feedback regarding the implementation of these models for precision agriculture applications. </p>
