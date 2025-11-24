import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, random_split
import torchvision
from tqdm import tqdm
from torch import optim
import copy
import argparse
import uuid
import json
from diffusers import AutoencoderKL, DDIMScheduler
import random
from unet import UNetModel
import wandb
from torchvision import transforms
from feature_extractor import ImageEncoder
from utils.iam_dataset import IAMDataset
from utils.GNHK_dataset import GNHK_Dataset
from utils.auxilary_functions import *
from torchvision.utils import save_image
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer
import cv2
import glob

torch.cuda.empty_cache()
OUTPUT_MAX_LEN = 95
IMG_WIDTH = 256
IMG_HEIGHT = 64

c_classes = '_!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
cdict = {c:i for i,c in enumerate(c_classes)}
icdict = {i:c for i,c in enumerate(c_classes)}

punctuation = [',', '.', '!', '?', ';', ':']

def save_images(images, path, args, **kwargs):
    grid = torchvision.utils.make_grid(images, padding=0, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
        if args.color == False:
            im = im.convert('L')
        else:
            im = im.convert('RGB')
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    im.save(path)
    return im

def save_single_images(image, path, args, **kwargs):
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(image)
        if args.color == False:
            im = im.convert('L')
        else:
            im = im.convert('RGB')
    else:
        ndarr = image.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    im.save(path)
    return im

def preprocess_custom_image(img_path, transform, args):
    """Preprocess custom handwriting images to match model expectations"""
    try:
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Resize to 64px height while maintaining aspect ratio
        (original_width, original_height) = img.size
        new_width = int(original_width * 64 / original_height)
        img_resized = img.resize((new_width, 64))
        
        # Pad width to 256px
        if new_width < 256:
            img_padded = ImageOps.pad(img_resized, size=(256, 64), color="white")
        else:
            # If too wide, gradually resize down
            current_width = new_width
            current_img = img_resized
            while current_width > 256:
                current_width = max(256, current_width - 20)
                current_img = current_img.resize((current_width, 64))
            img_padded = ImageOps.pad(current_img, size=(256, 64), color="white")
        
        # Convert to tensor and normalize
        img_tensor = transform(img_padded).to(args.device)
        return img_tensor
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def extract_style_from_custom_images(style_extractor, image_folder, transform, args):
    """Extract style features from custom handwriting images"""
    # Get all image files from folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(image_folder, ext.upper())))
    
    if not image_paths:
        raise ValueError(f"No images found in {image_folder}")
    
    print(f"Found {len(image_paths)} image files")
    
    # Preprocess all images
    style_tensors = []
    valid_image_paths = []
    for img_path in image_paths:
        tensor = preprocess_custom_image(img_path, transform, args)
        if tensor is not None:
            style_tensors.append(tensor)
            valid_image_paths.append(img_path)
    
    if not style_tensors:
        raise ValueError("No valid images could be processed")
    
    print(f"Successfully processed {len(style_tensors)} images")
    
    # Stack into batch and extract features
    style_batch = torch.stack(style_tensors)
    with torch.no_grad():
        style_features = style_extractor(style_batch)
    
    print(f"Style features shape: {style_features.shape}")
    
    # Average the features across all style images
    avg_style_features = style_features.mean(dim=0, keepdim=True)
    
    print(f"Average style features shape: {avg_style_features.shape}")
    print(f"Style features extracted from {len(style_tensors)} images")
    return avg_style_features

def remove_all_data_parallel_prefixes(state_dict):
    """Remove all DataParallel prefixes from state dict keys"""
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove all occurrences of 'module.' in the key
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

class PatchedUNetModel(UNetModel):
    """Patched UNet that can handle custom style features"""
    
    def forward(self, x, timesteps=None, context=None, y=None, original_images=None, mix_rate=None, style_extractor=None, **kwargs):
        """
        Forward pass with support for custom style features
        """
        # If we have custom style features, use them directly
        if style_extractor is not None and original_images is None:
            print(f"Using custom style features with shape: {style_extractor.shape}")
            # Store the custom style features for use in the UNet
            self.custom_style_features = style_extractor
            # Call parent forward but don't pass original_images to avoid the style extraction pipeline
            return super().forward(x, timesteps, context, y, **kwargs)
        else:
            # Use the original forward pass
            return super().forward(x, timesteps, context, y, original_images=original_images, mix_rate=mix_rate, style_extractor=style_extractor, **kwargs)

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(64, 256), args=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps).to(args.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.img_size = img_size
        self.device = args.device

    def sampling(self, model, vae, n, x_text, labels, args, style_extractor, noise_scheduler, custom_style_features=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None, run_idx=None):
        model.eval()
        
        with torch.no_grad():
            text_features = tokenizer(x_text, padding="max_length", truncation=True, return_tensors="pt", max_length=40).to(args.device)
            
            # Use custom style features if provided
            if custom_style_features is not None:
                print(f"Using custom style features with shape: {custom_style_features.shape}")
                # Pass the custom style features directly to the UNet
                style_features = custom_style_features.repeat(n, 1).to(args.device)
                style_images = None
            else:
                # Fallback to original method (using label-based style)
                print("Using label-based style (no custom features)")
                style_images = None
                style_features = None            
            
            if args.latent == True:
                x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
            else:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
            
            noise_scheduler.set_timesteps(50)
            for time in noise_scheduler.timesteps:
                t_item = time.item()
                t = (torch.ones(n) * t_item).long().to(args.device)

                with torch.no_grad():
                    # Pass style features through the style_extractor parameter
                    # The patched UNet will handle them appropriately
                    noisy_residual = model(x, t, text_features, labels, original_images=None, mix_rate=None, style_extractor=style_features)
                    prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
                    x = prev_noisy_sample

        model.train()
        if args.latent==True:
            latents = 1 / 0.18215 * x
            image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)
        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--img_size', type=int, default=(64, 256))  
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./diffusionpen_iam_model_path') 
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=False)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--sampling_word', type=bool, default=False) 
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--style_path', type=str, default='./style_models/iam_style_diffusionpen.pth')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--sampling_mode', type=str, default='single_sampling', help='single_sampling, paragraph')
    parser.add_argument('--custom_style_folder', type=str, default=None, help='Path to folder containing your handwriting samples')
    parser.add_argument('--text_to_generate', type=str, default='hello world', help='Text to generate in custom handwriting style')
    
    args = parser.parse_args()
    
    # Force CPU usage
    args.device = 'cpu'
    print('Starting sampling with custom handwriting style...')
    print(f'Using device: {args.device}')
    
    # Load models directly
    transform = transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load UNet - use the patched version
    if args.model_name == 'diffusionpen':
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        text_encoder = CanineModel.from_pretrained("google/canine-c")
        text_encoder = text_encoder.to(args.device)
    else:
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        text_encoder = None

    # Use the patched UNet instead of the original
    unet = PatchedUNetModel(
        image_size=args.img_size, in_channels=args.channels, model_channels=args.emb_dim,
        out_channels=args.channels, num_res_blocks=args.num_res_blocks, 
        attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=args.num_heads, 
        num_classes=339, context_dim=args.emb_dim, vocab_size=80, 
        text_encoder=text_encoder, args=args
    )
    
    unet = unet.to(args.device)
    
    # Load pre-trained weights - fix DataParallel issue
    print('Loading UNet weights...')
    checkpoint = torch.load(f'{args.save_path}/models/ckpt.pt', map_location=torch.device('cpu'))
    
    # Remove ALL DataParallel prefixes if they exist
    if any('module.' in key for key in checkpoint.keys()):
        print('Removing ALL DataParallel prefixes from state dict...')
        checkpoint = remove_all_data_parallel_prefixes(checkpoint)
    
    # Load the state dict with strict=False to ignore missing keys
    missing_keys, unexpected_keys = unet.load_state_dict(checkpoint, strict=False)
    
    if missing_keys:
        print(f"Missing keys (ignored): {len(missing_keys)}")
        # Print first few missing keys for debugging
        for i, key in enumerate(missing_keys[:5]):
            print(f"  - {key}")
        if len(missing_keys) > 5:
            print(f"  ... and {len(missing_keys) - 5} more")
    
    if unexpected_keys:
        print(f"Unexpected keys (ignored): {len(unexpected_keys)}")
        for i, key in enumerate(unexpected_keys[:5]):
            print(f"  - {key}")
        if len(unexpected_keys) > 5:
            print(f"  ... and {len(unexpected_keys) - 5} more")
    
    unet.eval()
    print('UNet model loaded')

    # Load VAE
    if args.latent:
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = vae.to(args.device)
        vae.requires_grad_(False)
        print('VAE loaded')
    else:
        vae = None

    # Load scheduler
    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
    
    # Load style extractor
    feature_extractor = ImageEncoder(model_name='mobilenetv2_100', num_classes=0, pretrained=True, trainable=True)
    state_dict = torch.load(args.style_path, map_location=torch.device('cpu'))
    
    # Remove DataParallel prefixes for style extractor too
    if any('module.' in key for key in state_dict.keys()):
        state_dict = remove_all_data_parallel_prefixes(state_dict)
    
    model_dict = feature_extractor.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(state_dict)
    feature_extractor.load_state_dict(model_dict)
    feature_extractor = feature_extractor.to(args.device)
    feature_extractor.requires_grad_(False)
    feature_extractor.eval()
    print('Style extractor loaded')

    # Initialize diffusion
    diffusion = Diffusion(img_size=args.img_size, args=args)

    # Extract custom style features if folder provided
    custom_style_features = None
    if args.custom_style_folder and os.path.exists(args.custom_style_folder):
        print(f"Extracting style from custom images in: {args.custom_style_folder}")
        custom_style_features = extract_style_from_custom_images(
            feature_extractor, args.custom_style_folder, transform, args
        )
        # Use a dummy label since we're using custom style
        labels = torch.tensor([0]).long().to(args.device)
        print("Using CUSTOM handwriting style")
    else:
        print("No custom style folder provided")
        # Fallback to random IAM style
        # style_id = random.randint(0, 338)
        style_id = 15  # Fixed style ID for consistency
        labels = torch.tensor([style_id]).long().to(args.device)
        custom_style_features = None
        print(f"Using random IAM style: {style_id}")

    # Generate text
    text_to_generate = args.text_to_generate
    print(f'Generating: "{text_to_generate}"')
    
    generated_images = diffusion.sampling(
        unet, vae, n=len(labels), x_text=text_to_generate, labels=labels, args=args,
        style_extractor=feature_extractor, noise_scheduler=ddim, 
        custom_style_features=custom_style_features,
        transform=transform, tokenizer=tokenizer, text_encoder=text_encoder
    )
    
    # Save results
    os.makedirs('./custom_generated', exist_ok=True)
    if custom_style_features is not None:
        output_path = f'./custom_generated/custom_style_{text_to_generate.replace(" ", "_")}.png'
    else:
        output_path = f'./custom_generated/random_style_{text_to_generate.replace(" ", "_")}.png'
    
    save_single_images(generated_images[0], output_path, args)
    print(f'Saved: {output_path}')

    print('Sampling completed!')

if __name__ == "__main__":
    main()