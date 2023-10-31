import os
import torch
from diffusers import StableDiffusionPipeline
from train.convert_to_original import hf_to_original
from train.fix_conversion import fix_diffusers_model_conversion


def convert_base_original_to_hf(config, base_model_name, ext=None, logger=None):
    data_base_dir = config['DATA_BASE_DIR']
    checkpoints_base_dir = f'{data_base_dir}/sd-models/models/Stable-diffusion'
    sd_config_file = f'{data_base_dir}/sd-configs/v1-inference.yaml'
    hf_pretrained_dir = f'{data_base_dir}/hf-pretrained'

    base_model_single_file = None
    exts = ('safetensors', 'ckpt') if ext is None else (ext,)
    for ext in exts:
        single_file = f'{checkpoints_base_dir}/{base_model_name}.{ext}'
        if os.path.exists(single_file):
            base_model_single_file = single_file
            break

    if base_model_single_file is None:
        if logger is not None:
            logger.warn('model file not found.')
        return False

    pipe = StableDiffusionPipeline.from_single_file(
        base_model_single_file,
        use_safetensors=True,
        load_safety_checker=False,
        local_files_only=True,
        dtype=torch.half,
        original_config_file=sd_config_file,
    )
    pipe.save_pretrained(f'{hf_pretrained_dir}/{base_model_name}')
    del pipe

    return True


def convert_trained_to_original(config, task_id, sub_dir=None):
    data_base_dir = config['DATA_BASE_DIR']
    if sub_dir == '_':
        sub_dir = None

    trains_dir = f'{data_base_dir}/trains'
    if sub_dir is not None:
        train_dir = f'{trains_dir}/{sub_dir}/t_{task_id}'
    else:
        train_dir = f'{trains_dir}/t_{task_id}'

    model_output_dir = f'{train_dir}/model'
    if not os.path.isdir(model_output_dir):
        return False

    model_file = f'{model_output_dir}/model.safetensors'
    hf_to_original(model_output_dir, model_file, half=True)
    fix_diffusers_model_conversion(model_file)
    return True
