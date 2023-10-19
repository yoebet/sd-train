import os
import signal
import subprocess
import time
import psutil
import pathlib


def build_args(args_map):
    args = []
    for k in args_map:
        v = args_map[k]
        if isinstance(v, str):
            args.append(f'--{k}={v}')
        elif isinstance(v, bool):
            if v:
                args.append(f'--{k}')
        elif isinstance(v, (int, float)):
            args.append(f'--{k}={v}')
        elif v is None:
            ...

    return args


def launch(config, task_params, train_params,
           accelerate_params=None,
           no_accelerate=False):
    data_base_dir = config['DATA_BASE_DIR']
    accelerate_config_file = config.get('ACCELERATE_CONFIG', None)
    launch_script_dir = config.get('LAUNCH_SCRIPT_DIR', '.')

    task_id = task_params.get('task_id', None)
    hf_local_files_only = task_params.get('hf_local_files_only', False)
    # user_id = task_params.get('user_id', None)
    # live/object/style
    # train_type = task_params.get('train_type', None)
    # base_model_name = task_params.get('base_model_name', None)
    # instance_images=task_params.get('instance_images', None)
    # device_index = task_params.get('device_index', None)

    if task_id is None:
        task_id = str(int(time.time()))
    train_params["task_id"] = task_id

    train_params['logging_dir'] = f'{data_base_dir}/logs/hot'
    train_dir = f'{data_base_dir}/trains/t_{task_id}'
    train_params['instance_data_dir'] = f'{train_dir}/instance_images'

    if train_params.get('with_prior_preservation'):
        train_params['class_data_dir'] = f'{train_dir}/class_images'

    output_dir = f'{train_dir}/output'
    train_params['output_dir'] = output_dir

    # pretrained_model_name_or_path

    train_args = build_args(train_params)

    if no_accelerate:
        args = ['nohup', 'python', f'{launch_script_dir}/train.py'] + train_args
    else:
        if accelerate_params is None:
            accelerate_args = []
        else:
            config_file = accelerate_params.pop('config_file')
            if config_file is not None:
                accelerate_config_file = config_file
            accelerate_args = build_args(accelerate_params)

        if accelerate_config_file is not None:
            acf = pathlib.Path(f'{data_base_dir}/accelerate-configs', accelerate_config_file)
            accelerate_args.insert(0, f'--config_file={str(acf)}')

        script_file = f'{launch_script_dir}/train.py'
        args = ['nohup', 'accelerate', 'launch'] + accelerate_args + [script_file] + train_args

    print(' '.join(args))

    # p = subprocess.Popen(args, preexec_fn=os.setpgrp)
    def preexec_function():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    p = subprocess.Popen(args,
                         preexec_fn=preexec_function,
                         env={'HF_HUB_OFFLINE': 'true' if hf_local_files_only else ''})

    return p.pid


if __name__ == "__main__":
    config = {
        'DATA_BASE_DIR': 'data'
    }
    task_params = {
        'task_id': 'etrew',
    }
    train_params = {
        'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
        'instance_prompt': 'a photo of sks dog',
    }
    accelerate_params = {
        'config_file': 'c1.yaml'
    }
    pid = launch(config, task_params, train_params, accelerate_params)
    print(pid)
    pinfo = psutil.Process(pid).as_dict()
    pinfo.pop('environ')
    print(pinfo)

    # time.sleep(10)
    # pinfo = psutil.Process(pid).as_dict()
    # pinfo.pop('environ')
    # print(pinfo)

    time.sleep(1000)
