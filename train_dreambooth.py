#!/usr/bin/env python
# coding=utf-8

import copy
import gc
import itertools
import logging
import math
import os
import shutil
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import model_info
from packaging import version
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from train.class_images import gen_class_images
from train.datasets import DreamBoothDataset, collate_fn
from train.hf_repo import put_to_hf
from train.train_args import parse_args
from train.validate_and_test import log_validation, log_test, log_instance_images
from train.convert_to_original import hf_to_original
from train.fix_conversion import fix_diffusers_model_conversion
from train.checkpointing import try_resume_from_checkpoint, save_checkpoint
from train.text_encoding import import_text_encoder_class, encode_prompt, pre_compute_text_embeddings
from train.train_epochs import train_epochs

# print(os.environ)


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__)


def model_has_vae(args):
    config_file_name = os.path.join("vae", AutoencoderKL.config_name)
    if os.path.isdir(args.pretrained_model_name_or_path):
        config_file_name = os.path.join(args.pretrained_model_name_or_path, config_file_name)
        return os.path.isfile(config_file_name)
    else:
        files_in_repo = model_info(args.pretrained_model_name_or_path, revision=args.revision).siblings
        return any(file.rfilename == config_file_name for file in files_in_repo)


def main(args):
    args.model_output_dir = str(Path(args.output_dir, 'model'))
    args.checkpoints_dir = str(Path(args.output_dir, 'checkpoints'))
    args.test_output_dir = str(Path(args.output_dir, 'test'))
    args.validations_dir = str(Path(args.output_dir, 'validations'))

    if args.validation_prompt is None or args.validation_prompt == '':
        args.validation_prompt = args.instance_prompt

    if args.test_prompts_file is not None and os.path.isfile(args.test_prompts_file):
        test_prompts = json.load(open(args.test_prompts_file))
    else:
        logger.info(f'args.test_prompts_file not configured')
        test_prompts = [
            {'prompt': args.instance_prompt,
             'negative_prompt': ''}
        ]
    args.custom_pipeline = "lpw_stable_diffusion"

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir,
                                                      logging_dir=args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(args)
    args.test_prompts = test_prompts

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.pretrained_model_name_or_path is not None and args.pretrained_model_name_or_path != '':
        pass
    elif args.base_model_single_file is not None:
        load_params = {
            'use_safetensors': True,
            'load_safety_checker': False,
            'local_files_only': True,
            'dtype': torch.half,
            'original_config_file': args.base_model_config_file,
        }
        if args.base_model_single_file.endswith('.safetensors'):
            tmp_model_file = str(Path(args.output_dir, Path(args.base_model_single_file).name))
            fix_diffusers_model_conversion(args.base_model_single_file, tmp_model_file)
            pipe = StableDiffusionPipeline.from_single_file(
                tmp_model_file,
                **load_params,
            )
            os.remove(tmp_model_file)
        else:
            pipe = StableDiffusionPipeline.from_single_file(
                args.base_model_single_file,
                **load_params,
            )
        hf_pretrained_dir = args.hf_pretrained_dir
        pretrained_model_path = f'{hf_pretrained_dir}/{args.base_model_name}'
        pipe.save_pretrained(pretrained_model_path)
        del pipe
        logger.info(f'[base_model] save pretrained: {pretrained_model_path}')

        args.pretrained_model_name_or_path = pretrained_model_path

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        gen_class_images(args, accelerator, logger=logger)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.model_output_dir, exist_ok=True)
        os.makedirs(args.checkpoints_dir, exist_ok=True)
        if os.path.exists(args.validations_dir):
            shutil.rmtree(args.validations_dir)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
    else:
        tokenizer = None

    # import correct text encoder class
    text_encoder_cls = import_text_encoder_class(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    # noise_scheduler = DDIMScheduler.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     subfolder="scheduler",
    # )
    # noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(
    #     # f'{args.hf_alt_dir}',
    #     args.pretrained_model_name_or_path,
    #     subfolder="scheduler",
    # )
    # noise_scheduler.set_timesteps(30)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    if model_has_vae(args):
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
        )
        logger.info('model has vae.')
    else:
        logger.info('model not has vae.')
        vae = None

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    # unet.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, model_output_dir):
        if accelerator.is_main_process:
            for model in models:
                sub_dir = "unet" if isinstance(model, type(accelerator.unwrap_model(unet))) else "text_encoder"
                model.save_pretrained(os.path.join(model_output_dir, sub_dir))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                # load transformers style into model
                load_model = text_encoder_cls.from_pretrained(
                    input_dir,
                    subfolder="text_encoder",
                )
                model.config = load_model.config
            else:
                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir,
                    subfolder="unet",
                )
                model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if vae is not None:
        vae.requires_grad_(False)

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.xformers_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if args.train_text_encoder and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    train_te_separately = args.train_text_encoder and args.train_te_separately and args.learning_rate_te is not None

    # Optimizer creation
    if not args.train_text_encoder or train_te_separately:
        params_to_optimize = unet.parameters()
    else:
        params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters())
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if train_te_separately:
        optimizer_te = optimizer_class(
            text_encoder.parameters(),
            lr=args.learning_rate_te,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer_te = None

    if args.pre_compute_text_embeddings:
        (pre_computed_encoder_hidden_states,
         validation_prompt_embeds,
         validation_negative_prompt_embeds,
         class_prompt_encoder_hidden_states) = pre_compute_text_embeddings(args,
                                                                           text_encoder,
                                                                           tokenizer)
        text_encoder = None
        tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()
    else:
        pre_computed_encoder_hidden_states = None
        validation_prompt_embeds = None
        validation_negative_prompt_embeds = None
        class_prompt_encoder_hidden_states = None

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        class_num=args.num_class_images,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        encoder_hidden_states=pre_computed_encoder_hidden_states,
        class_prompt_encoder_hidden_states=class_prompt_encoder_hidden_states,
        tokenizer_max_length=args.tokenizer_max_length,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    if train_te_separately:
        lr_scheduler_te = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_te,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )
    else:
        lr_scheduler_te = None

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        text_encoder = accelerator.prepare(text_encoder)
        if train_te_separately:
            optimizer_te, lr_scheduler_te = accelerator.prepare(
                optimizer_te, lr_scheduler_te
            )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)

    if not args.train_text_encoder and text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        # tracker_config = vars(copy.deepcopy(args))
        # tracker_config.pop("validation_images")
        track_keys = ['base_model_name',
                      'num_train_epochs', 'max_train_steps',
                      'train_batch_size', 'gradient_accumulation_steps', 'offset_noise',
                      'learning_rate', 'learning_rate_te', 'train_text_encoder_ratio',
                      'lr_scheduler', 'lr_warmup_steps', 'lr_power', 'lr_num_cycles',
                      'adam_beta1', 'adam_beta2', 'adam_weight_decay', 'max_grad_norm',
                      'num_class_images', 'prior_loss_weight',
                      'validation_scheduler', 'mixed_precision']
        tracker_config = {k: v for k, v in (vars(args)).items() if k in track_keys}
        proj_name = f't_{args.task_id}' if args.task_id else 'dreambooth'
        accelerator.init_trackers(proj_name, config=tracker_config)

    if accelerator.is_main_process:
        instance_images = train_dataset.load_all_instance_images()
        log_instance_images(accelerator, instance_images)
        del instance_images

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        global_step, first_epoch = try_resume_from_checkpoint(args,
                                                              accelerator,
                                                              num_update_steps_per_epoch)
    else:
        global_step, first_epoch = 0, 0

    train_epochs_kwargs = {'accelerator': accelerator, 'args': args,
                           'first_epoch': first_epoch, 'global_step': global_step,
                           'train_dataloader': train_dataloader, 'unet': unet, 'vae': vae,
                           'tokenizer': tokenizer, 'text_encoder': text_encoder,
                           'noise_scheduler': noise_scheduler, 'weight_dtype': weight_dtype,
                           'validation_prompt_embeds': validation_prompt_embeds,
                           'validation_negative_prompt_embeds': validation_negative_prompt_embeds,
                           'logger': logger
                           }
    if train_te_separately:
        max_train_steps = args.max_train_steps
        if args.max_train_te_steps is not None:
            max_train_steps = args.max_train_te_steps
        train_epochs(train_unet=False, train_te=True,
                     optimizer=optimizer_te, lr_scheduler=lr_scheduler_te,
                     max_train_steps=max_train_steps,
                     **train_epochs_kwargs)
        del optimizer_te
        del lr_scheduler_te
        logger.info(f'done training text encoder')

    train_epochs(train_unet=True,
                 train_te=args.train_text_encoder and not train_te_separately,
                 optimizer=optimizer, lr_scheduler=lr_scheduler,
                 max_train_steps=args.max_train_steps,
                 **train_epochs_kwargs)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline_args = {}

        if text_encoder is not None:
            pipeline_args["text_encoder"] = accelerator.unwrap_model(text_encoder)

        if args.skip_save_text_encoder:
            pipeline_args["text_encoder"] = None

        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            noise_scheduler=noise_scheduler,
            revision=args.revision,
            custom_pipeline=args.custom_pipeline,
            **pipeline_args,
        )
        # pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2)

        # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
        scheduler_args = {}

        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)

        pipeline.save_pretrained(args.model_output_dir)

        with open(f'{args.model_output_dir}/token_identifier.txt', 'w') as f:
            f.write(args.instance_prompt)
        logger.info(f'done save_pretrained.')

        image_args_list = log_test(pipeline, args, accelerator, global_step=args.max_train_steps, logger=logger)
        logger.info(f'done log_test.')

        model_file = f'{args.model_output_dir}/model.safetensors'
        hf_to_original(args.model_output_dir, model_file, use_safetensors=True, half=True)
        logger.info(f'done conversion.')
        fix_diffusers_model_conversion(model_file)
        logger.info(f'done fix conversion.')

        if args.push_to_hub:
            put_to_hf(args, pipeline=pipeline, image_args_list=image_args_list)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
