import os
import re
from pprint import pformat
import base64
import shutil
import torch
from flask import Flask, jsonify, request, Response, abort, send_file
from dotenv import dotenv_values
from launch import launch
from prepare import prepare_instance_images
from conversion import convert_base_original_to_hf, convert_trained_to_original
from train.dirs import get_train_dir
from model_sync import sync_file

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


def trans_unit(bytes, unit):
    if unit is None:
        return bytes
    k = 1024
    div = {'B': 1, 'K': k, 'M': k * k, 'G': k * k * k}.get(unit.upper())
    return bytes / div


@app.route('/check_mem_all/available', methods=('GET',))
def check_mem_all():
    unit = request.args.get('unit')
    import accelerate
    d = accelerate.utils.get_max_memory()
    pairs = [(i, trans_unit(n, unit)) for i, n in d.items()]
    return jsonify(pairs)


@app.route('/check_mem/<device_index>', methods=('GET',))
def check_device_mem(device_index):
    device_index = int(device_index)
    unit = request.args.get('unit')
    free, total = torch.cuda.mem_get_info(device_index)
    return jsonify({
        'free': trans_unit(free, unit),
        'total': trans_unit(total, unit),
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
    logger.info(pformat(req))
    task = req.get('task')
    launch_options = req.get('launch')
    grouped_train_params = req.get('train')

    if launch_options is None:
        launch_options = {}

    config = app.config

    try:
        skip_download_ie = launch_options.get('skip_download_dataset_if_exists')
        prepare_res = prepare_instance_images(config,
                                              task,
                                              skip_if_exists=skip_download_ie,
                                              logger=logger)
    except Exception as e:
        logger.error(e)
        return jsonify({
            'success': False,
            'error_message': f"[prepare] {type(e)}: {e}"
        })

    if task['task_id'] is None:
        task['task_id'] = prepare_res.get('task_id')

    device_index = launch_options.get('device_index', None)
    if device_index is not None:
        free, total = torch.cuda.mem_get_info(device_index)
        occupied = total - free
        k = 1024
        if occupied > 6 * k * k * k:
            return (jsonify({
                'success': False,
                'error_message': 'device occupied',
            }))

    train_params = {}

    # grouped -> flatten
    for k in grouped_train_params:
        p = grouped_train_params[k]
        if isinstance(p, dict):
            train_params.update(p)
        else:
            train_params[k] = p

    try:
        # logger.info(train_params)
        result = launch(config,
                        task,
                        launch_options,
                        train_params,
                        logger=logger)
    except Exception as e:
        logger.error(e)
        return jsonify({
            'success': False,
            'error_message': f"[launch] {type(e)}: {e}"
        })

    return jsonify(result)


def check_train_dirs(task_id, dir_names, sub_dir=None):
    data_base_dir = app.config['DATA_BASE_DIR']
    train_dir = get_train_dir(data_base_dir, task_id, sub_dir=sub_dir)
    if not os.path.exists(train_dir):
        return False
    return all([os.path.exists(f'{train_dir}/{dir}') for dir in dir_names])


@app.route('/task/<task_id>/status', methods=('POST',))
def check_task_status(task_id):
    req = request.get_json()
    pid = req.get('root_pid')
    pid = int(pid)
    sub_dir = req.get('sub_dir')

    import psutil

    running = False

    if psutil.pid_exists(pid):
        rp = psutil.Process(pid)
        pname = rp.name()
        logger.info(pname)
        if 'accelerate' not in pname and 'python' not in pname:
            # raise Exception(f'wrong pid: {pid}, {pname}')
            return jsonify({
                'success': True,
                'task_status': 'failed',
                'failure_reason': 'wpn'
            })
        try:
            rp.cmdline()
        except psutil.ZombieProcess:
            pass
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

    dir_ok = check_train_dirs(task_id, ['test'], sub_dir)
    if dir_ok:
        return jsonify({
            'success': True,
            'task_status': 'done',
        })
    else:
        return jsonify({
            'success': True,
            'task_status': 'failed',
            'failure_reason': 'ntest'
        })


@app.route('/task/<task_id>/<sub_dir>/generate_single', methods=('POST',))
def generate_single(task_id, sub_dir):
    success = convert_trained_to_original(app.config,
                                          task_id,
                                          sub_dir)
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


@app.route('/task/<task_id>/<sub_dir>/test_images', methods=('GET',))
def list_test_images(task_id, sub_dir):
    data_base_dir = app.config['DATA_BASE_DIR']
    train_dir = get_train_dir(data_base_dir, task_id, sub_dir=sub_dir)
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


@app.route('/task/<task_id>/<sub_dir>/test_images/<filename>', methods=('GET',))
def get_test_image_file(task_id, sub_dir, filename):
    data_base_dir = app.config['DATA_BASE_DIR']
    train_dir = get_train_dir(data_base_dir, task_id, sub_dir=sub_dir)
    image_path = f'{train_dir}/test/{filename}'
    return send_file(image_path)


@app.route('/task/<task_id>/release', methods=('POST',))
def release_model(task_id):
    req = request.get_json()
    sub_dir = req.get('sub_dir', None)
    target_model_name = req.get('target_model_name')
    if target_model_name is None:
        base_model_name = req.get('base_model_name')
        if base_model_name is not None:
            target_model_name = f'{base_model_name}-t_{task_id}'
        else:
            target_model_name = f't_{task_id}'

    data_base_dir = app.config['DATA_BASE_DIR']
    checkpoints_base_dir = f'{data_base_dir}/sd-models/models/Stable-diffusion'

    train_dir = get_train_dir(data_base_dir, task_id, sub_dir=sub_dir)

    model_file = f'{train_dir}/model/model.safetensors'
    if not os.path.isfile(model_file):
        return jsonify({
            'success': False,
            'error_message': 'no model file'
        })

    target_file_name = f'{target_model_name}.safetensors'
    target_model_file = f'{checkpoints_base_dir}/{target_file_name}'
    existed = os.path.isfile(target_model_file)
    shutil.copyfile(model_file, target_model_file)
    if existed:
        logger.warning(f'model file overwritten: {target_file_name}')

    try:
        sync_file(data_base_dir, target_file_name, logger=logger)
    except Exception as e:
        logger.error(e)

    return jsonify({
        'success': True
    })


@app.route('/task/<task_id>/undo-release', methods=('POST',))
def undo_release_model(task_id):
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

    target_file_name = f'{target_model_name}.safetensors'
    target_model_file = f'{checkpoints_base_dir}/{target_file_name}'
    if os.path.isfile(target_model_file):
        os.remove(target_model_file)
        logger.info(f'target file removed: {target_file_name}')

    return jsonify({
        'success': True
    })


@app.route('/sync-checkpoint', methods=('POST',))
def sync_model():
    req = request.get_json()
    data_base_dir = app.config['DATA_BASE_DIR']
    target_file_name = req.get('model_file')
    if '.' not in target_file_name:
        target_file_name = f'{target_file_name}.safetensors'

    try:
        pid = sync_file(data_base_dir, target_file_name, logger=logger)
    except Exception as e:
        logger.error(e)
        return jsonify({
            'success': False,
            'error_message': str(e)
        })

    return jsonify({
        'success': True,
        'pid': pid
    })


tokenizer = None


@app.route('/check-tokens', methods=('POST',))
def check_tokens():
    req = request.get_json()
    text = req.get('text', None)
    if text is None:
        raise Exception('missing `text` parameter')

    data_base_dir = app.config['DATA_BASE_DIR']
    global tokenizer
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            f'{data_base_dir}/hf-alt/tokenizer',
            local_files_only=True
        )
    t = tokenizer
    tokens = [t.decode(c) for c in t.encode(text)][1:-1]
    return jsonify({
        'count': len(tokens),
        'parts': tokens,
        'split': '|'.join(tokens)
    })


def get():
    return app


if __name__ == '__main__':
    app.run()
