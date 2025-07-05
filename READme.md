# Gibbs Diffusion for Blind Denoising

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) <!-- Assuming you'll use MIT -->

## Description
This project is associated with the submission of the Final Research Project as part of the MPhil in Data Intensive Science at the University of Cambridge. The associated project report (this thesis) can be found under `report/DIS_thesis.pdf`. The associated executive summary can be found under `report/DIS_executive_summary.pdf`.

The primary objective of this project is to reproduce the results presented in "Listening to the Noise: Blind Denoising with Gibbs Diffusion" by Heurtel-Depeiges et al. (2024) [https://arxiv.org/pdf/2402.19455]. This work involves implementing the Gibbs Diffusion (GDiff) algorithm and validating its performance on tasks such as blind denoising of natural images with colored noise and cosmological parameter inference from simulated CMB data. The study aims to understand the mechanisms, evaluate performance, identify limitations, and analyze the GDiff framework.

We further extended it to perform a comparative study against baseline models such as DnCNN and BM3D for the natural image denoising application.

## Table of Contents
- [Data Availability](#data-availability)
- [Installation](#installation)
- [Usage](#usage)
- [Reproduced Features](#reproduced-features)
- [Support](#support)
- [License](#license)
- [Documentation](#documentation)
- [Project Status](#project-status)
- [Author and Acknowledgments](#author-and-acknowledgments)
- [Note on the Use of Auto-generation Tools](#note-on-the-use-of-auto-generation-tools)

## Data Availability

The primary datasets used for reproducing the results of the GDiff paper are publicly available:
- **Image Denoising:**
    - Training: Tiny-ImageNet-200 (a subset of ImageNet) dataset ([http://cs231n.stanford.edu/tiny-imagenet-200.zip](http://cs231n.stanford.edu/tiny-imagenet-200.zip)).
    - Evaluation: A held-out set of Tiny-ImageNet-200 images.
- **Cosmological Parameter Inference:**
    - Dust Maps: Derived from simulations available in the CATS database ([https://www.catalogue-of-astrophysical-turbulence-simulations.org/](https://www.catalogue-of-astrophysical-turbulence-simulations.org/)).
    - CMB Maps: Generated using CAMB ([https://camb.info/](https://camb.info/)) based on cosmological parameters.

## Installation

To set up the environment for reproducing this project, follow these steps:

### Requirements

- Python 3.9 or higher.
- Conda (for managing the Python environment).
- PyTorch (version used in GDiff or compatible, e.g., 1.13+).
- Standard scientific Python libraries: NumPy, SciPy, Matplotlib.
- For cosmological application: `camb`, `pixell`, `astropy`.
- Docker (optional, if providing a Dockerized environment).

### Setup

#### Local Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/am3353.git
    cd am3353
    ```
2.  **Set up the Python Environment:**
    Install packages using `pip install -r requirements.txt`.
3.  **Download Data**
    Create the data storage directory `./data/`
    
    <b> For Natural Images </b>: To download the Tiny-ImageNet-200 dataset, run 
    
    ```python -m modules.utils.download_tiny_imagenet``` 
    
    This will automatically download and unzip all related files into the `./data/` directory.
    
    <b> For Cosmological Inference </b>: To download the necessary files, first run 
    
    ```python -m modules.utils.download_cosmo_data``` 
    
    This will download Cho-ENO simulations for Mach ~7 at different timesteps (t_500, t_600, t_650, t_700, t_750) into the `./data/cosmo/` directory. 
    
    hen, to create random N(=3000 by default) 256x256 dust-maps, cmb-maps, and mixed-maps, run
    
    ```python -m modules.utils.cosmo_create``` 
    
    This will generate N instances stored in `./data/cosmo/created_data/`   

## Usage

This project primarily involves running Python scripts and Jupyter notebooks to reproduce the experiments from the GDiff paper.

### Model Training (py-scripts) 

Due to the inability to save and upload the models to the GitLab repository (because of their large size), we provide scripts to directly train all three models (including the 1D model) using the command:

```python -m modules.main_run --mode=1D/2D/cosmo```

The trained models will be saved in the `./saves/` directory for proper loading in the IPython notebooks. The saved models should follow this structure:

- `saves/gdiffusion_1d_model.pt`: Conditional diffusion model for 1D signal denoising.
- `saves/gdiffusion_2d_model.pt`: Conditional diffusion model for Image denoising.
- `saves/gdiffusion_cosmo_model.pt.pt`: Conditional diffusion model for cosmology application.

### Scripts & Notebooks

Key scripts and notebooks are located in the HOME directory, which detail the benchmark vaniall/standard denoising, 1D + 2D natural images non-blind and blind denoising and the cosmological CMB segregation and parameter inferencce:

-   `unit_main_denoise.ipynb`: Runs vanilla denoising algorithms (DAE, DnCNN, BM3D) for benchmarking reasons and performance understanding reasons, quantitative results in PSNR, SSIM and L1 metrics recorded.
-   `unit_main_gibbs.ipynb`: Runs the GDiff algorithm for blind denoising of natural images (+ 1D signals), generates denoised images, infers noise parameters, and produces quantitative results (PSNR, SSIM, L1).
-   `unit_main_cosmo.ipynb`: Applies GDiff to simulated CMB data for component separation and cosmological parameter inference, generating reconstructed maps, power spectra, and parameter posteriors.

##### Utility Scripts

```plaintext
- modules/comp/: Gibbs-Diff model components.
    └── ../one_d/*.py   # Components for the 1D Gibbs-Diff model architecture.
    └── ../two_d/*.py   # Components for 2D image and cosmological model architectures.

- modules/utils/*.py    # Utility scripts for model implementation.
```

##### Running Experiments:

1.  **Image Denoising:**
    Open and run the `unit_main_gibbs.ipynb` notebook. Load the trained models, and set the input/output directories as needed within the notebook. Run the required cells.

2.  **Cosmology Inference:**
    Open and run the `unit_main_cosmo.ipynb` notebook. Load the trained models, and set the input/output directories as needed within the notebook. Run the required cells.


## Reproduced Features

This project aims to reproduce the following key features and results from Heurtel-Depeiges et al. (2024):
-   **Blind Denoising of Natural Images:**
    -   Qualitative image reconstruction for various colored noises.
    -   Quantitative PSNR/SSIM/L1 performance.
    -   Power Spectra deviation of the reconstruction.
    -   Posterior inference for noise amplitude ($\sigma$) and spectral index ($\bar{\phi}$).
-   **Cosmological Parameter Inference:**
    -   Blind separation of dust and CMB components from mixed observations.
    -   Posterior inference for cosmological parameters ($H_0, \omega_b, \sigma_{\text{CMB}}$).
    -   Recovery of component power spectra.

-   Demonstration of the Gibbs sampling loop combining diffusion models and HMC.

## Support
For questions regarding this reproduction study, please contact Alik Mondal at [am3353@cam.ac.uk](mailto:am3353@cam.ac.uk).
For questions regarding the original GDiff paper, please refer to the contact information provided by its authors.

## License
This project code, developed for the MPhil reproduction study, is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Note that the original GDiff code (if publicly available) and datasets used may have their own licenses.

## Documentation
Code documentation is provided within the Python scripts.
For a detailed explanation of the methodology, please refer to:
-   This thesis: `report/DIS_thesis.pdf`.
-   The original GDiff paper: Heurtel-Depeiges, D., Margossian, C. C., Ohana, R., & Régaldo-Saint Blancard, B. (2024). Listening to the Noise: Blind Denoising with Gibbs Diffusion. *ICML*. (or arXiv link).

## Project Status
This project, focusing on the reproduction of GDiff, is complete for the MPhil submission. All planned reproduction experiments have been conducted, and results are documented in the associated thesis report.

## Author and Acknowledgments
This reproduction study was conducted by Alik Mondal as part of the MPhil in Data Intensive Science at the University of Cambridge.
Acknowledgments to the supervisors Boris Bolliet and Fiona McCarthy for their guidance.
Acknowledgments to the authors of "Listening to the Noise: Blind Denoising with Gibbs Diffusion" for their foundational work.

## Note on the Use of Auto-generation Tools
This section details the use of AI-powered auto-generation tools in the development of this project and its documentation.

### Google Gemini Pro  
Google Gemini Pro was used extensively during the development phase for:

- **Code templating and scaffolding**, especially in the early stages of implementing complex modules.  
- **Refactoring**, to improve modularity, readability, and structure of existing code.  
- **Debugging assistance**, where Gemini was helpful in critically analyzing error sources and suggesting fault lines in logic or numerical instability.

In particular, Gemini was instrumental in:

- Designing the initial **Hamiltonian Monte Carlo (HMC) sampler templates** (`hmc.py` and `hmc_v2.py`).  
- Incorporating advanced features such as **adaptive step size adjustment** and **inverse mass matrix adaptation** into the samplers.

All Gemini-generated suggestions were reviewed and adapted to fit the project’s specific requirements and design philosophy.

---

### ChatGPT (OpenAI)  
ChatGPT was consulted selectively, primarily for:

- **Grammar correction and language refinement** in the project report.  
- **Code beautification and complexity reduction**, where existing implementations were rewritten for clarity and performance.

Examples include:

- **Prompt 1 (Code Refactoring):** "Simplify this nested loop structure in PyTorch for batched matrix operations."  
  - *ChatGPT Output (Summary):* Provided vectorized alternatives and PyTorch idioms to eliminate inefficient loop constructs.  
  - *Modification/Use in Project:* Resulted in faster and cleaner implementations in the inference module.

- **Prompt 2 (Report Language Polishing):** "Rewrite this paragraph for academic tone and clarity."  
  - *ChatGPT Output (Summary):* Offered grammatically improved and more concise formulations.  
  - *Modification/Use in Report:* Applied throughout Sections 2, 3, and 5 to ensure professional tone and coherence.

The use of these tools was aimed at improving efficiency and exploring different ways to present information or solve technical challenges. The final intellectual content, analysis, and conclusions presented in this thesis are the author's own.
```