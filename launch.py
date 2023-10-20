import os
import signal
import subprocess
import time
import psutil
import pathlib
import logging

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


def launch(config, task, launch_options, train_params):
    data_base_dir = config['DATA_BASE_DIR']
    accelerate_config_file = config.get('ACCELERATE_CONFIG', None)
    launch_script_dir = config.get('LAUNCH_SCRIPT_DIR', '.')

    hf_hub_offline = launch_options.get('hf_hub_offline', False)
    shell = launch_options.get('shell', False)
    wrap_proxy = launch_options.get('wrap_proxy', False)
    hf_accelerate = launch_options.get('hf_accelerate', False)
    accelerate_params = launch_options.get('accelerate', None)
    proxy_command = launch_options.get('proxy_command', 'proxychains4')

    # user_id = task_params.get('user_id', None)
    # live/object/style
    # train_type = task_params.get('train_type', None)
    # base_model_name = task_params.get('base_model_name', None)
    # instance_images=task_params.get('instance_images', None)
    # device_index = task_params.get('device_index', None)

    task_id = task.get('task_id', None)
    if task_id is None:
        task_id = str(int(time.time()))
    train_params["task_id"] = task_id

    train_params['logging_dir'] = f'{data_base_dir}/logs/hot'
    train_dir = f'{data_base_dir}/trains/t_{task_id}'
    train_params['instance_data_dir'] = f'{train_dir}/instance_images'

    if train_params.get('with_prior_preservation'):
        train_params['class_data_dir'] = f'{train_dir}/class_images'

    train_params['output_dir'] = train_dir
    log_file = f'{train_dir}/log-{str(int(time.time()))}.txt'

    # pretrained_model_name_or_path

    train_args = build_args(train_params, shell=shell)
    script_file = f'{launch_script_dir}/train.py'

    if hf_accelerate:
        if accelerate_params is None:
            accelerate_args = []
        else:
            config_file = accelerate_params.get('config_file', None)
            if config_file is not None:
                accelerate_config_file = config_file
            accelerate_args = build_args(accelerate_params, shell=shell)

        if accelerate_config_file is not None:
            acf = pathlib.Path(f'{data_base_dir}/accelerate-configs', accelerate_config_file)
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
        logging.info(cmd)
        p = subprocess.Popen(cmd,
                             preexec_fn=preexec_function,
                             env=env,
                             shell=True)
    else:
        if wrap_proxy:
            args = proxy_command.split() + args
        logging.info(' '.join(args))
        log_file_h = open(log_file, 'x')
        p = subprocess.Popen(args,
                             preexec_fn=preexec_function,
                             start_new_session=True,
                             env=env,
                             stdout=log_file_h,
                             stderr=subprocess.STDOUT)

    logging.info(f'log file: {log_file}')

    return p.pid


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
    pid = launch(config, task, launch_options, train_params)
    print(pid)
    pinfo = psutil.Process(pid).as_dict()
    pinfo.pop('environ')
    print(pinfo)

    # time.sleep(10)
    # pinfo = psutil.Process(pid).as_dict()
    # pinfo.pop('environ')
    # print(pinfo)

    time.sleep(1000)
