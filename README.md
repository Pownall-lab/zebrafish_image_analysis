# zebrafish_image_analysis
Image analysis pipelines for quantitative analysis of zebrafish embryo ChromExM and diffraction-limited microscopy datasets

This repository contains image-analysis pipelines for quantitative analysis of zebrafish embryo ChromExM and diffraction-limited microscopy datasets.

The repository currently contains two primary workflows:

1. Unexpanded image analysis
   - Cellpose-based nuclear segmentation
   - Intensity quantification
   - Statistical analysis


2. Expanded image analysis
   - U-Net-based segmentation
   - Chromatin occupancy analysis
   - Autocorrelation analysis
  
Trained models are available at https://zenodo.org/records/21923246
#Model applicability: 

These models were trained and validated using images generated in the Pownall Lab with a single imaging system and specific acquisition settings, sample preparations, and labeling conditions. Performance has not been systematically evaluated on data acquired using other instruments or experimental conditions.

Users should assess segmentation quality on representative images from their own datasets before using model outputs for quantitative analysis. Depending on differences in imaging system, image resolution, signal-to-noise ratio, staining, sample preparation, or acquisition settings, additional preprocessing, parameter adjustment, or model fine-tuning may be required.

Automated segmentation results should be visually inspected and validated for the intended application.

## Environment setup

The analysis was developed using Python 3.11. Create and activate a virtual environment from the repository directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

Install PyTorch and torchvision:

```bash
python -m pip install torch torchvision
```

For NVIDIA GPU support, use the installation command recommended for your system by the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

GPU acceleration is recommended for Cellpose segmentation and U-Net training and inference, although other analysis steps can run on a CPU.

## Tested configuration

The code was tested with:

- Linux
- Python 3.11.2
- PyTorch 2.11.0 with CUDA 13.0
- torchvision 0.26.0
- Cellpose 4.0.5
- 64-core AMD Ryzen Threadripper
- 256 GB RAM
- NVIDIA GeForce RTX 4090 with 24 GB VRAM
- NVIDIA RTX PRO 5000 Blackwell with 48 GB VRAM

These specifications describe the system used for development and analysis; they are not minimum requirements. The scripts normally use one GPU per process rather than combining the memory of both GPUs.

Systems with less memory may require a smaller training batch size, fewer parallel workers, or processing fewer images simultaneously.
