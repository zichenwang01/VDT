import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from models import VDT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from mask_generator import VideoMaskGenerator
from utils import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    # General settings
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--display", action="store_true", help="Display images")
    # Model settings
    parser.add_argument("--model", type=str, choices=list(VDT_models.keys()), default="VDT-L/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--ckpt", type=str, default="checkpoints/model.pt",
                        help="Optional path to a VDT checkpoint.")
    # Diffusion settings
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--num-sampling-steps", type=int, default=16)  # Set higher for better results! (max 1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    # Experiment settings
    parser.add_argument('--task', type=str, default=None, 
                        choices=['predict', 'backward', 'interpolation', 'unconditional', 
                                'one_frame', 'arbitrary_interpolation', 'spatial_temporal'],
                        help="Task type for video processing")
    parser.add_argument('--ratio', type=float, default=0.5,
                        help="Spatial mask ratio for spatially varying task")
    parser.add_argument('--input_video', type=str, default="data/test/cloud.pt",
                        help="Path to input video tensor file")
    parser.add_argument('--output_dir', type=str, default=f'./exp',
                        help="Directory to save output images")
    parser.add_argument('--name', type=str, default=f'{time.strftime("%Y-%m-%d_%H-%M-%S")}',
                        help="Name of the experiment")
    
    return parser.parse_args()

def setup_model(args):
    # Setup PyTorch
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load model
    latent_size = args.image_size // 8
    additional_kwargs = {'num_frames': args.num_frames, 'mode': 'video'} 
    model = VDT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        **additional_kwargs
    )

    model, _ = load_checkpoint(model, args.ckpt)
    model = model.to(device)
    model.eval()  # important!
    
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    return model, diffusion, vae, latent_size

def load_input_video(args, device):
    x = torch.load(args.input_video).to(device)
    B, T, C, H, W = x.shape
    print(f"Input video shape: {x.shape}") 
    
    # x = x[0].unsqueeze(0)  # Select first video and add batch dimension
    # print(f"Input video shape: {x.shape}") 
    # torch.save(x, f"cloud.pt")
    # input("Press Enter to continue...")
    
    # Reshape for visualization
    x_flat = x.view(-1, C, H, W).to(device=device) # B*T, C, H, W
    save_image(x_flat, f"{args.output_dir}/input.png", nrow=16, normalize=True, value_range=(-1, 1))
    
    # Optionally display the input video frames
    if args.display:
        img = mpimg.imread(f"{args.output_dir}/input.png")
        plt.figure(figsize=(10, 8), dpi=500)
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(f"{args.output_dir}/input_display.png")
        plt.close()
    
    return x, x_flat, B

def get_task_choice(args):
    choice_map = {
        'predict': 0,
        'backward': 1,
        'interpolation': 2,
        'unconditional': 3,
        'one_frame': 4,
        'arbitrary_interpolation': 5,
        'spatial_temporal': 6,
        'spatial_varying': 7
    }
    
    if args.task is None:
        # Interactive mode if no task is specified via command line
        choice_name = input("Please select task type (predict, backward, interpolation, unconditional, one_frame, arbitrary_interpolation, spatial_temporal, spatial_varying): ")
        return choice_map[choice_name]
    else:
        return choice_map[args.task]

def process_video(args, model, diffusion, vae, latent_size, x, raw_x, B, task_idx):
    # Encode input to latent space
    with torch.no_grad():
        x_latent = vae.encode(raw_x).latent_dist.sample().mul_(0.18215)
    x_latent = x_latent.view(-1, args.num_frames, 4, x_latent.shape[-2], x_latent.shape[-1])
    print(f"Latent shape: {x_latent.shape}") # B, T, 4, latent_size, latent_size
    print(f"Latent size: {latent_size}")
    
    # Initialize random noise
    z = torch.randn(B, args.num_frames, 4, latent_size, latent_size, device=device)
    
    # Create mask for the selected task
    generator = VideoMaskGenerator(
        input_size=(x_latent.shape[-4], x_latent.shape[-2], x_latent.shape[-1]),
        spatial_mask_ratio=args.ratio,
    )
    mask = generator(B, device, idx=task_idx)
    
    # Prepare for sampling
    sample_fn = model.forward
    z = z.permute(0, 2, 1, 3, 4)
    
    # Run diffusion sampling
    samples = diffusion.p_sample_loop(
        sample_fn, z.shape, z, clip_denoised=False, progress=True, device=device,
        raw_x=x_latent, mask=mask
    )
    
    # Process samples
    samples = samples.permute(1, 0, 2, 3, 4) * mask + x_latent.permute(2, 0, 1, 3, 4) * (1-mask)
    samples = samples.permute(1, 2, 0, 3, 4)  # 4, 16, 8, 32, 32 -> 16 8 4
    samples = samples.reshape(-1, 4, latent_size, latent_size) / 0.18215
    
    # Decode latents in chunks to avoid OOM
    decoded_chunks = []
    chunk_size = 256
    num_chunks = (samples.shape[0] + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, samples.shape[0])
        chunk = samples[start_idx:end_idx]
        
        decoded_chunk = vae.decode(chunk).sample
        decoded_chunks.append(decoded_chunk)
    
    samples = torch.cat(decoded_chunks, dim=0)
    samples = samples.reshape(-1, args.num_frames, samples.shape[-3], samples.shape[-2], samples.shape[-1])
    
    # Process mask for visualization
    mask = F.interpolate(mask.float(), size=(raw_x.shape[-2], raw_x.shape[-1]), mode='nearest')
    mask = mask.unsqueeze(0).repeat(3, 1, 1, 1, 1).permute(1, 2, 0, 3, 4) 
    
    # Process original input for visualization
    raw_x_reshaped = raw_x.reshape(-1, args.num_frames, raw_x.shape[-3], raw_x.shape[-2], raw_x.shape[-1])
    raw_x_masked = raw_x_reshaped * (1 - mask)
    
    # Concatenate input and output for visualization
    final_output = torch.cat([raw_x_masked, samples], dim=1)
    print(f"Output shape: {final_output.shape}")
    
    # Save output images
    save_image(
        final_output.reshape(-1, final_output.shape[-3], final_output.shape[-2], final_output.shape[-1]), 
        f"{args.output_dir}/output.png", 
        nrow=16, 
        normalize=True, 
        value_range=(-1, 1)
    )
    
    # Optionally display output
    if args.display:
        img = mpimg.imread(f"{args.output_dir}/output.png")
        plt.figure(figsize=(10, 8), dpi=500)
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(f"{args.output_dir}/output_display.png")
        plt.close()
    
    return final_output

def main():
    # Parse arguments
    args = parse_args()
    
    # Make output directory
    args.output_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup model components
    model, diffusion, vae, latent_size = setup_model(args)
    
    # Load and process input video
    x, raw_x, B = load_input_video(args, device)
    
    # Get task choice
    task_idx = get_task_choice(args)
    
    # Process video with the selected task
    final_output = process_video(args, model, diffusion, vae, latent_size, x, raw_x, B, task_idx)
    
    print("Processing complete!")
    return final_output

if __name__ == "__main__":
    main()