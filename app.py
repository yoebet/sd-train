from flask import Flask, jsonify, request, Response, abort
from dotenv import dotenv_values
from launch import launch
from prepare import prepare_instance_images

app = Flask(__name__)

app.config.from_mapping(dotenv_values())


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
    return jsonify(res)


@app.route('/task/launch', methods=('POST',))
def launch_task():
    config = app.config
    req = request.get_json()
    app.logger.info(req)
    task = req.get('task')
    launch_options = req.get('launch')
    train_params = req.get('train')

    if launch_options is None:
        launch_options = {}

    skip_download_dataset_if_exists = launch_options.get('skip_download_dataset_if_exists')

    prepare_res = prepare_instance_images(config,
                                          task,
                                          skip_if_exists=skip_download_dataset_if_exists)
    if task['task_id'] is None:
        task['task_id'] = prepare_res.get('task_id')

    # grouped -> flatten
    for k in train_params:
        p = train_params[k]
        if isinstance(p, dict):
            train_params.pop(k)
            train_params.update(p)

    app.logger.info(train_params)
    result = launch(config,
                    task,
                    launch_options,
                    train_params)

    return jsonify(result)


@app.route('/task/<id>', methods=('GET',))
def task_status(id):
    return jsonify({'id': id,
                    'status': 'ok',
                    })


def get():
    return app


if __name__ == '__main__':
    app.run()
