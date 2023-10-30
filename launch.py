import os
import signal
import subprocess
import time
import psutil
import pathlib
import logging
import hashlib

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def build_args(args_map, shell=False):
    args = []
    for k in args_map:
        v = args_map[k]
        if isinstance(v, str):
            if shell:
                if ' ' in v or '"' in v:
                    v = v.replace('"', '\\"')
                    args.append(f'--{k}="{v}"')
                else:
                    args.append(f'--{k}={v}')
            else:
                args.append(f'--{k}={v}')
        elif isinstance(v, bool):
            if v:
                args.append(f'--{k}')
        elif isinstance(v, (int, float)):
            args.append(f'--{k}={v}')
        elif v is None:
            ...

    return args


def locate_base_model(train_params, data_base_dir, logger=None):
    base_model_name = train_params.get('base_model_name', None)
    if base_model_name is None:
        raise Exception('missing base_model_name')
    base_model_file_name = train_params.pop('base_model_file_name', None)
    hf_repo_id = train_params.pop('hf_repo_id', None)
    hf_repo_first = train_params.pop('hf_repo_first', False)

    hf_pretrained_dir = f'{data_base_dir}/hf-pretrained'
    train_params['hf_pretrained_dir'] = hf_pretrained_dir

    if hf_repo_id is not None and hf_repo_first:
        train_params['pretrained_model_name_or_path'] = hf_repo_id
        return

    pretrained_base_model = f'{hf_pretrained_dir}/{base_model_name}'
    if os.path.isdir(pretrained_base_model):
        train_params['pretrained_model_name_or_path'] = pretrained_base_model
        return

    sd_config_file = f'{data_base_dir}/sd-configs/v1-inference.yaml'
    if os.path.exists(sd_config_file):
        train_params['base_model_config_file'] = sd_config_file

    checkpoints_base_dir = f'{data_base_dir}/sd-models/models/Stable-diffusion'

    if base_model_file_name is not None:
        base_model_single_file = f'{checkpoints_base_dir}/{base_model_file_name}'
        if os.path.exists(base_model_single_file):
            train_params['base_model_single_file'] = base_model_single_file
            return
        if logger is not None:
            logger.warn(f'base model file not found: {base_model_file_name}')

    for ext in ('safetensors', 'ckpt'):
        base_model_single_file = f'{checkpoints_base_dir}/{base_model_name}.{ext}'
        if os.path.exists(base_model_single_file):
            train_params['base_model_single_file'] = base_model_single_file
            return


def determine_class_data_dir(train_params, data_base_dir, logger=None):
    base_model_name = train_params.get('base_model_name', None)
    if not train_params.get('with_prior_preservation', False):
        return
    if train_params.get('class_data_dir') is not None:
        return
    class_prompt = train_params.get('class_prompt')
    p_hash = hashlib.md5(class_prompt.encode('utf8')).hexdigest()
    class_data_dir = f'{data_base_dir}/class-images/{base_model_name}--{p_hash}'
    train_params['class_data_dir'] = class_data_dir
    if os.path.isdir(class_data_dir):
        return
    os.makedirs(class_data_dir, exist_ok=True)
    with open(f'{class_data_dir}/_meta.txt', 'x') as f:
        f.write(f'prompt: {class_prompt}\n')


def launch(config, task, launch_options, train_params, logger=None):
    if logger is None:
        logger = logging.getLogger('launch')

    data_base_dir = config['DATA_BASE_DIR']
    accelerate_config_file = config.get('ACCELERATE_CONFIG', None)
    launch_script_dir = config.get('LAUNCH_SCRIPT_DIR', '.')

    hf_hub_offline = launch_options.get('hf_hub_offline', False)
    shell = launch_options.get('shell', False)
    wrap_proxy = launch_options.get('wrap_proxy', False)
    hf_accelerate = launch_options.get('hf_accelerate', False)
    accelerate_params = launch_options.get('accelerate', None)
    proxy_command = launch_options.get('proxy_command', 'proxychains4 -q')

    task_id = task.get('task_id', None)
    if task_id is None:
        task_id = str(int(time.time()))
    train_params["task_id"] = task_id

    locate_base_model(train_params, data_base_dir, logger=logger)

    train_type = task.get('train_type', None)
    if train_type == 'live':
        test_prompts_file = 'train/test_prompts_live.json'
    else:
        test_prompts_file = 'train/test_prompts_object.json'
    train_params['test_prompts_file'] = test_prompts_file

    if train_params.get('center_crop', None) is None:
        train_params['center_crop'] = train_type != 'live'

    train_params['logging_dir'] = f'{data_base_dir}/logs/hot'
    train_dir = f'{data_base_dir}/trains/t_{task_id}'
    train_params['instance_data_dir'] = f'{train_dir}/instance_images'

    determine_class_data_dir(train_params, data_base_dir, logger=logger)

    train_params['output_dir'] = train_dir
    log_file = f'{train_dir}/log-{str(int(time.time()))}.txt'

    # with open(f'{train_dir}/_meta.txt', 'w') as f:
    #     f.write(f'class_prompt: {class_prompt}\n')

    train_args = build_args(train_params, shell=shell)
    script_file = f'{launch_script_dir}/train_dreambooth.py'

    if hf_accelerate:
        if accelerate_params is None:
            accelerate_args = []
        else:
            config_file = accelerate_params.pop('config_file', None)
            if config_file is not None:
                accelerate_config_file = config_file
            accelerate_args = build_args(accelerate_params, shell=shell)

        if accelerate_config_file is not None:
            acf = pathlib.Path(f'{data_base_dir}/hf-accelerate', accelerate_config_file)
            accelerate_args.insert(0, f'--config_file={str(acf)}')

        args = ['accelerate', 'launch'] + accelerate_args + [script_file] + train_args
    else:
        args = ['python', script_file] + train_args

    # p = subprocess.Popen(args, preexec_fn=os.setpgrp)
    def preexec_function():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    # https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
    env = os.environ.copy()
    env['HF_HUB_OFFLINE'] = 'true' if hf_hub_offline else ''

    if shell:
        cmd = ' '.join(args)
        if wrap_proxy:
            cmd = f'{proxy_command} {cmd}'
        cmd = f'nohup {cmd} > {log_file} 2>&1'
        logger.info(cmd)
        p = subprocess.Popen(cmd,
                             preexec_fn=preexec_function,
                             env=env,
                             shell=True)
    else:
        if wrap_proxy:
            args = proxy_command.split() + args
        logger.info(' '.join(args))
        log_file_h = open(log_file, 'x')
        p = subprocess.Popen(args,
                             preexec_fn=preexec_function,
                             start_new_session=True,
                             env=env,
                             stdout=log_file_h,
                             stderr=subprocess.STDOUT)

    logger.info(f'log file: {log_file}')

    return {'success': True, 'root_pid': p.pid, 'task_id': task_id}


if __name__ == "__main__":
    config = {
        'DATA_BASE_DIR': 'data'
    }
    task = {
        'task_id': 'etrew',
    }
    launch_options = {
        'hf_accelerate': True,
        'accelerate': {
            'config_file': 'c1.yaml'
        }
    }
    train_params = {
        'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
        'instance_prompt': 'a photo of sks dog',
    }
    result = launch(config, task, launch_options, train_params)
    pid = result.get('pid')
    print(pid)
    pinfo = psutil.Process(pid).as_dict()
    pinfo.pop('environ')
    print(pinfo)

    # time.sleep(10)
    # pinfo = psutil.Process(pid).as_dict()
    # pinfo.pop('environ')
    # print(pinfo)

    time.sleep(1000)
