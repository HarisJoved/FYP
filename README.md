# DiffusionPen - Handwriting Generation with Diffusion Models

DiffusionPen is a deep learning project that generates realistic handwriting using diffusion models. It supports both custom handwriting styles and pre-trained IAM dataset styles.

## Features

- 🎨 **Custom Handwriting Style**: Extract style features from your own handwriting samples
- 📝 **Text-to-Handwriting**: Generate any text in a specified handwriting style
- 🎯 **Pre-trained Models**: Use pre-trained models trained on IAM dataset
- 🔄 **Flexible Generation**: Support for both latent space and pixel space generation

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended) or CPU
- PyTorch

### Setup

1. Clone the repository:ash
git clone <your-repo-url>
cd DiffusionPen2. Install dependencies:
pip install -r requirements.txt3. Download pre-trained models:
   - Place UNet checkpoint in `./diffusionpen_iam_model_path/models/ckpt.pt`
   - Place style extractor in `./style_models/iam_style_diffusionpen.pth`

## Quick Start

### Generate Handwriting with Custom Style

The easiest way to generate handwriting with your own style:

python generate_handwriting.py --text_to_generate "your text here"This script uses reference images from `imgs/IAM_Words_Images/` to extract style features.

### Generate with Custom Style Folder

For more control, use `trainCopy.py`:

python trainCopy.py \
    --custom_style_folder ./path/to/your/handwriting/images \
    --text_to_generate "hello world" \
    --device cuda
**Parameters:**
- `--custom_style_folder`: Path to folder containing your handwriting sample images (PNG, JPG, etc.)
- `--text_to_generate`: Text you want to generate
- `--device`: `cuda` or `cpu` (default: `cpu`)
- `--save_path`: Path to UNet model checkpoint (default: `./diffusionpen_iam_model_path`)
- `--style_path`: Path to style extractor model (default: `./style_models/iam_style_diffusionpen.pth`)

### Generate with Pre-trained IAM Styles

To use pre-trained IAM dataset styles:

python trainCopy.py \
    --text_to_generate "hello world" \
    --device cuda(Omitting `--custom_style_folder` will use a random IAM style)

## Project Structure
