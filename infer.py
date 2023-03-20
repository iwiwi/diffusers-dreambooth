import argparse
import os
import re
import time
from diffusers import DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from transformers import CLIPTextModel
import torch

def generate_images(pipeline_id, model_path, checkpoint_step, num_images, output_dir, prompt, negative_prompt, batch_size):
    if checkpoint_step is not None:
        unet = UNet2DConditionModel.from_pretrained(f"{model_path}/checkpoint-{checkpoint_step}/unet")
        text_encoder = CLIPTextModel.from_pretrained(f"{model_path}/checkpoint-{checkpoint_step}/text_encoder")
        pipeline = DiffusionPipeline.from_pretrained(pipeline_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.to("cuda")

    # Create a timestamped subdirectory with an escaped prompt
    timestamp = time.strftime("%y%m%d-%H%M%S")
    escaped_prompt = re.sub(r'[<>:"/\|?*]', '_', prompt)[:100]
    output_subdir = os.path.join(output_dir, f"{timestamp} {escaped_prompt}")
    os.makedirs(output_subdir, exist_ok=True)

    image_count = 0
    for i in range((num_images + batch_size - 1) // batch_size):
        images = pipeline(prompt, num_inference_steps=100, guidance_scale=7.5, negative_prompt=negative_prompt, num_images_per_prompt=batch_size).images
        for j, image in enumerate(images):
            image.save(f"{output_subdir}/image_{image_count}.png")
            image_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from a specified model ID and checkpoint step.")
    parser.add_argument("--pipeline_id", type=str, default="runwayml/stable-diffusion-v1-5", help="Pipeline ID to use for inference (default: 'runwayml/stable-diffusion-v1-5').")
    parser.add_argument("--model_path", type=str, required=True, help="Model ID to use for inference.")
    parser.add_argument("--checkpoint_step", type=int, default=None, help="Checkpoint step to use for inference (optional).")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate (default: 1).")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for generated images (default: 'output').")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to use for generating images.")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt not to guide the image generation.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of images to generate per prompt (default: 1).")
    args = parser.parse_args()

    generate_images(args.pipeline_id, args.model_path, args.checkpoint_step, args.num_images, args.output_dir, args.prompt, args.negative_prompt, args.batch_size)


