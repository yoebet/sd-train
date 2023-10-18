import os
import io
import importlib
import numpy as np
import torch
import shutil
import hashlib
from diffusers import DiffusionPipeline
from PIL import Image
from diffusers.utils import is_wandb_available

if is_wandb_available():
    import wandb


def log_validation(
        text_encoder,
        tokenizer,
        unet,
        vae,
        args,
        accelerator,
        weight_dtype,
        global_step,
        prompt_embeds,
        negative_prompt_embeds,
        logger,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )

    pipeline_args = {}

    if vae is not None:
        pipeline_args["vae"] = vae

    if text_encoder is not None:
        text_encoder = accelerator.unwrap_model(text_encoder)

    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=accelerator.unwrap_model(unet),
        revision=args.revision,
        torch_dtype=weight_dtype,
        **pipeline_args,
    )

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    module = importlib.import_module("diffusers")
    scheduler_class = getattr(module, args.validation_scheduler)
    pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config, **scheduler_args)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.pre_compute_text_embeddings:
        pipeline_args = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }
    else:
        pipeline_args = {"prompt": args.validation_prompt}

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    if args.validation_images is None:
        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(**pipeline_args, num_inference_steps=25, generator=generator).images[0]
            images.append(image)
    else:
        for image in args.validation_images:
            image = Image.open(image)
            image = pipeline(**pipeline_args, image=image, generator=generator).images[0]
            images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, global_step, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()

    return images


def log_test(
        pipeline,
        args,
        accelerator,
        global_step,
        logger,
):
    logger.info(
        f"Running test... \n Generating {args.num_test_images} images with prompt:"
        f" {args.validation_prompt}."
    )

    test_output_dir = args.test_output_dir
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True)

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type
        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"
        scheduler_args["variance_type"] = variance_type

    module = importlib.import_module("diffusers")
    scheduler_class = getattr(module, args.validation_scheduler)
    pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config, **scheduler_args)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    pipeline_args = {"prompt": args.instance_prompt}

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    for i in range(args.num_test_images):
        with torch.autocast("cuda"):
            image = pipeline(**pipeline_args, num_inference_steps=50, generator=generator).images[0]
        images.append(image)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_hash = hashlib.sha1(img_byte_arr.getvalue()).hexdigest()
        file_name = f'{test_output_dir}/{i + 1}-{image_hash}.png'
        image.save(file_name)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("test", np_images, global_step, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "test": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    torch.cuda.empty_cache()

    return images
