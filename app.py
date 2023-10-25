import os
from flask import Flask, jsonify, request, Response, abort
from dotenv import dotenv_values
from launch import launch
from prepare import prepare_instance_images

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


@app.route('/task/prepare', methods=('POST',))
def prepare_task():
    req = request.get_json()
    task = req.get('task')
    res = prepare_instance_images(app.config, task)
    # base_model_name = req.get('base_model_name', None)
    return jsonify(res)


@app.route('/task/launch', methods=('POST',))
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


@app.route('/task/status', methods=('POST',))
def check_task_status():
    req = request.get_json()
    task_id = req.get('task_id')
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


def get():
    return app


if __name__ == '__main__':
    app.run()
