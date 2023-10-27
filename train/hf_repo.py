import os
from pathlib import Path
from huggingface_hub import create_repo, model_info, upload_folder
from diffusers import DiffusionPipeline, StableDiffusionPipeline


def save_model_card(
        repo_id: str,
        image_args_list=None,
        base_model=str,
        train_text_encoder=False,
        prompt=str,
        repo_folder=None,
        pipeline: DiffusionPipeline = None,
):
    img_str = ""
    for i, (image, gen_args) in enumerate(image_args_list):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
instance_prompt: {prompt}
tags:
- {'stable-diffusion' if isinstance(pipeline, StableDiffusionPipeline) else 'if'}
- {'stable-diffusion-diffusers' if isinstance(pipeline, StableDiffusionPipeline) else 'if-diffusers'}
- text-to-image
- diffusers
- dreambooth
inference: true
---
    """
    model_card = f"""
# DreamBooth - {repo_id}

This is a dreambooth model derived from {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/).
You can find some example images in the following. \n
{img_str}

DreamBooth for the text encoder was enabled: {train_text_encoder}.
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def put_to_hf(args, pipeline, image_args_list):
    repo_id = args.hub_model_id
    if repo_id is None:
        if args.task_id is not None:
            repo_id = f'{args.base_model_name}-t_{args.task_id}'
        else:
            out_dir = Path(args.model_output_dir)
            repo_id = out_dir.name
            if repo_id == 'model':
                repo_id = out_dir.parent.name
    repo_id = create_repo(
        repo_id=repo_id,
        exist_ok=True,
        token=args.hub_token
    ).repo_id

    if args.pretrained_model_name_or_path and args.pretrained_model_name_or_path.count('/') <= 1:
        base_model = args.pretrained_model_name_or_path
    else:
        base_model = args.base_model_name
    save_model_card(
        repo_id,
        image_args_list=image_args_list,
        base_model=base_model,
        train_text_encoder=args.train_text_encoder,
        prompt=args.instance_prompt,
        repo_folder=args.model_output_dir,
        pipeline=pipeline,
    )
    upload_folder(
        repo_id=repo_id,
        folder_path=args.model_output_dir,
        commit_message="End of training",
        ignore_patterns=["step_*", "epoch_*"],
    )
