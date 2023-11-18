import os
import io
import json
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
        f"Running validation... \n Generating {args.num_validation_images} images."
    )
    test_prompts = args.test_prompts
    n_prompts = len(test_prompts)

    output_dir = f'{args.validations_dir}/s_{global_step}'
    os.makedirs(output_dir, exist_ok=True)

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
        custom_pipeline=args.custom_pipeline,
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
        for i in range(args.num_validation_images):
            if not args.pre_compute_text_embeddings:
                test_args = test_prompts[i % n_prompts]
                prompt = test_args.get('prompt')
                prompt = f'{args.instance_prompt}, {prompt}'
                negative_prompt = test_args.get('negative_prompt')
                pipeline_args = {"prompt": prompt,
                                 "negative_prompt": negative_prompt,
                                 }
            with torch.autocast("cuda"):
                image = pipeline(**pipeline_args, num_inference_steps=50, generator=generator).images[0]
            images.append(image)
    else:
        for image in args.validation_images:
            image = Image.open(image)
            image = pipeline(**pipeline_args, image=image, generator=generator).images[0]
            images.append(image)
    for i, image in enumerate(images):
        image.save(f'{output_dir}/{i + 1}.png')

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, global_step, dataformats="NHWC")
        elif tracker.name == "wandb":
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
        f"Running test... \n Generating {args.num_test_images} images."
    )

    test_output_dir = args.test_output_dir
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True)

    test_prompts = args.test_prompts
    n_prompts = len(test_prompts)
    logger.info(f'test prompts: {n_prompts}')

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

    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    image_args_list = []
    for i in range(args.num_test_images):
        test_args = test_prompts[i % n_prompts]
        prompt = test_args.get('prompt')
        prompt = f'{args.instance_prompt}, {prompt}'
        negative_prompt = test_args.get('negative_prompt')
        pipeline_args = {"prompt": prompt,
                         "negative_prompt": negative_prompt,
                         }

        if i < n_prompts:
            logger.info(f'test ({i + 1}): {pipeline_args}')
        elif i == n_prompts:
            logger.info(f'test ({i + 1}): ...')
        with torch.autocast("cuda"):
            image = pipeline(**pipeline_args, num_inference_steps=50, generator=generator).images[0]
        image_args_list.append((image, pipeline_args))
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_hash = hashlib.sha1(img_byte_arr.getvalue()).hexdigest()[:16]
        file_name = f'{test_output_dir}/{i + 1}-{image_hash}.png'
        image.save(file_name)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(image_args[0]) for image_args in image_args_list])
            tracker.writer.add_images("test", np_images, global_step, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "test": [
                        wandb.Image(img,
                                    caption=f"{i + 1}: {gen_args.get('prompt')}")
                        for i, (img, gen_args) in enumerate(image_args_list)
                    ]
                }
            )

    torch.cuda.empty_cache()

    return image_args_list


def log_instance_images(accelerator, images):
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for i, img in enumerate(images):
                np_image = np.asarray(img)
                tracker.writer.add_image("dataset", np_image, i, dataformats="HWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "dataset": [
                        wandb.Image(img, caption=f"{i + 1}")
                        for i, img in enumerate(images)
                    ]
                }
            )
