# generate_handwriting.py  ←  FINAL WORKING SCRIPT
import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineTokenizer, CanineModel
from feature_extractor import ImageEncoder
from unet import UNetModel
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--text_to_generate', type=str, default='world', help='Text to generate in custom handwriting style')

args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def preprocess(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    new_w = int(w * 64 / h)
    img = img.resize((new_w, 64), Image.LANCZOS)
    if new_w > 256: img = img.resize((256, 64), Image.LANCZOS)
    if new_w < 256: img = ImageOps.pad(img, (256, 64), color="white")
    return transform(img)

# Load models
tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
text_encoder = CanineModel.from_pretrained("google/canine-c").to(DEVICE)

class Args:
    interpolation = False
    img_feat = False
    latent = True
    color = True
    mix_rate = None
    device = DEVICE

unet = UNetModel(
    image_size=(64, 256),
    in_channels=4,
    model_channels=320,
    out_channels=4,
    num_res_blocks=1,
    attention_resolutions=(1,1),
    channel_mult=(1, 1),
    num_heads=4,
    num_classes=339,
    context_dim=320,
    vocab_size=80,
    text_encoder=text_encoder,
    args=Args()
).to(DEVICE)

# Load weights
state = torch.load("./diffusionpen_iam_model_path/models/ckpt.pt", map_location=DEVICE)
if any(k.startswith("module.") for k in state.keys()):
    state = {k.replace("module.", ""): v for k, v in state.items()}
unet.load_state_dict(state, strict=False)
unet.eval()

vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(DEVICE)
scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
scheduler.set_timesteps(50)

# Style extractor
style_extractor = ImageEncoder(model_name='mobilenetv2_100', num_classes=0, pretrained=True, trainable=True)
style_state = torch.load("./style_models/iam_style_diffusionpen.pth", map_location=DEVICE)
if any(k.startswith("module.") for k in style_state.keys()):
    style_state = {k.replace("module.", ""): v for k, v in style_state.items()}
style_extractor.load_state_dict(style_state, strict=False)
style_extractor.to(DEVICE).eval()

# Load your images
paths = glob.glob("imgs/IAM_Words_Images/*.png") + glob.glob("imgs/IAM_Words_Images/*.jpg")
paths = [p for p in paths if os.path.isfile(p)]
print(f"Found {len(paths)} reference images, using up to 100")
ref_images = torch.stack([preprocess(p).to(DEVICE) for p in paths[:100]])

# Text
text_input = tokenizer(args.text_to_generate, padding="max_length", truncation=True, max_length=40, return_tensors="pt").to(DEVICE)

# Sampling
latents = torch.randn((1, 4, 8, 32), device=DEVICE)
for t in scheduler.timesteps:
    t_tensor = torch.tensor([t], device=DEVICE)
    with torch.no_grad():
        pred = unet(
            x=latents,
            timesteps=t_tensor,
            context=text_input,
            y=torch.zeros(1, dtype=torch.long, device=DEVICE),
            original_images=ref_images,
            style_extractor_model=style_extractor   # This triggers the fix
        )
    latents = scheduler.step(pred, t, latents).prev_sample

# Decode
with torch.no_grad():
    latents = latents / 0.18215
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].cpu().permute(1, 2, 0).numpy()
    Image.fromarray((image * 255).astype("uint8")).save("MY_HANDWRITING.png")
    print("SUCCESS! Saved as MY_HANDWRITING.png")