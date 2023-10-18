import os
import re
import base64
import shutil
import torch
from flask import Flask, jsonify, request, Response, abort
from dotenv import dotenv_values
from launch import launch
from prepare import prepare_instance_images
from conversion import convert_base_original_to_hf, convert_trained_to_original

app = Flask(__name__)

app.config.from_mapping(dotenv_values())
app.config.from_mapping(dotenv_values('.env.local'))

logger = app.logger


@app.route('/', methods=('GET',))
def index():
    return 'ok'


@app.before_request
def before_request_callback():
    path = request.path
    if path != '/':
        auth = request.headers.get('AUTHORIZATION')
        if not auth == app.config['AUTHORIZATION']:
            abort(400)


@app.route('/check_mem_all/available', methods=('GET',))
def check_mem_all():
    import accelerate
    return accelerate.utils.get_max_memory()


def trans_unit(bytes, unit):
    if unit is None:
        return bytes
    k = 1024
    div = {'B': 1, 'K': k, 'M': k * k, 'G': k * k * k}.get(unit.upper())
    return bytes / div


@app.route('/check_mem/<device_index>', methods=('GET',))
def check_device_mem(device_index):
    device_index = int(device_index)
    unit = request.args.get('unit')
    total = torch.cuda.get_device_properties(device_index).total_memory
    reserved = torch.cuda.memory_reserved(device_index)
    allocated = torch.cuda.memory_allocated(device_index)
    return jsonify({
        'total': trans_unit(total, unit),
        'reserved': trans_unit(reserved, unit),
        'allocated': trans_unit(allocated, unit)
    })


@app.route('/prepare_task', methods=('POST',))
def prepare_task():
    req = request.get_json()
    task = req.get('task')
    res = prepare_instance_images(app.config, task)
    # base_model_name = req.get('base_model_name', None)
    return jsonify(res)


@app.route('/launch', methods=('POST',))
def launch_task():
    req = request.get_json()
    logger.info(req)
    task = req.get('task')
    launch_options = req.get('launch')
    train_params = req.get('train')

    if launch_options is None:
        launch_options = {}

    config = app.config

    skip_download_ie = launch_options.get('skip_download_dataset_if_exists')
    prepare_res = prepare_instance_images(config,
                                          task,
                                          skip_if_exists=skip_download_ie,
                                          logger=logger)
    if task['task_id'] is None:
        task['task_id'] = prepare_res.get('task_id')

    # grouped -> flatten
    for k in train_params:
        p = train_params[k]
        if isinstance(p, dict):
            train_params.pop(k)
            train_params.update(p)

    logger.info(train_params)
    result = launch(config,
                    task,
                    launch_options,
                    train_params,
                    logger=logger)

    return jsonify(result)


def check_train_dirs(task_id, dir_names):
    data_base_dir = app.config['DATA_BASE_DIR']
    train_dir = f'{data_base_dir}/trains/t_{task_id}'
    if not os.path.exists(train_dir):
        return False
    return all([os.path.exists(f'{train_dir}/{dir}') for dir in dir_names])


@app.route('/task/<task_id>/status', methods=('POST',))
def check_task_status(task_id):
    req = request.get_json()
    pid = req.get('root_pid')
    pid = int(pid)

    import psutil

    running = False

    if psutil.pid_exists(pid):
        rp = psutil.Process(pid)
        # Python
        logger.info(rp.name())
        try:
            logger.info(rp.cmdline())
        except psutil.ZombieProcess:
            return jsonify({
                'success': True,
                'task_status': 'failed',
                'failure_reason': 'unknown'
            })
        except psutil.AccessDenied:
            logger.error('AccessDenied')
            return jsonify({
                'success': False,
                'error_message': 'wrong pid'
            })

        pstatus = rp.status()
        if pstatus == 'running':
            running = True
        elif pstatus == 'zombie':
            pass
        else:
            logger.info(f'process status: {pstatus}')
            running = True

    if running:
        return jsonify({
            'success': True,
            'task_status': 'running',
        })

    dir_ok = check_train_dirs(task_id, ['test'])
    if dir_ok:
        return jsonify({
            'success': True,
            'task_status': 'done',
        })
    else:
        return jsonify({
            'success': True,
            'task_status': 'failed',
            'failure_reason': 'unknown'
        })


@app.route('/task/<task_id>/generate_single', methods=('POST',))
def generate_single(task_id):
    success = convert_trained_to_original(app.config,
                                          task_id)
    return jsonify({
        'success': success
    })


@app.route('/prepare_base_hf', methods=('POST',))
def prepare_base_hf():
    req = request.get_json()
    base_model_name = req.get('base_model_name')
    ext = req.get('ext')
    success = convert_base_original_to_hf(app.config,
                                          base_model_name,
                                          ext,
                                          logger=logger)
    return jsonify({
        'success': success
    })


@app.route('/task/<task_id>/test_images', methods=('GET',))
def list_test_images(task_id):
    data_base_dir = app.config['DATA_BASE_DIR']
    train_dir = f'{data_base_dir}/trains/t_{task_id}'
    test_output_dir = f'{train_dir}/test'
    if not os.path.isdir(test_output_dir):
        return jsonify({
            'success': False,
            'error_message': 'test output dir not exists'
        })

    files = os.listdir(test_output_dir)
    files = [f for f in files if re.match(r'\d+-', f)]
    return jsonify({
        'success': True,
        'filenames': files
    })


@app.route('/task/<task_id>/test_images/<filename>/b64', methods=('GET',))
def get_test_image(task_id, filename):
    data_base_dir = app.config['DATA_BASE_DIR']
    train_dir = f'{data_base_dir}/trains/t_{task_id}'
    image_path = f'{train_dir}/test/{filename}'
    if not os.path.isfile(image_path):
        return jsonify({
            'success': False,
            'error_message': 'no such file'
        })

    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
    return jsonify({
        'success': True,
        'base64': encoded_image
    })


@app.route('/task/<task_id>/release', methods=('POST',))
def release_model(task_id):
    req = request.get_json()
    target_model_name = req.get('target_model_name')
    if target_model_name is None:
        base_model_name = req.get('base_model_name')
        if base_model_name is not None:
            target_model_name = f'{base_model_name}-t_{task_id}'
        else:
            target_model_name = f't_{task_id}'

    data_base_dir = app.config['DATA_BASE_DIR']
    checkpoints_base_dir = f'{data_base_dir}/sd-models/models/Stable-diffusion'

    model_file = f'{data_base_dir}/trains/t_{task_id}/model/model.safetensors'
    if not os.path.isfile(model_file):
        return jsonify({
            'success': False,
            'error_message': 'no model file'
        })

    target_file_name = f'{target_model_name}.safetensors'
    target_model_file = f'{checkpoints_base_dir}/{target_file_name}'
    if os.path.isfile(target_model_file):
        logger.warning(f'target file exists: {target_file_name}')

    shutil.copyfile(model_file, target_model_file)

    return jsonify({
        'success': True
    })


def get():
    return app


if __name__ == '__main__':
    app.run()
