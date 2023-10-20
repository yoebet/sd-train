import logging
from flask import Flask, jsonify, request, Response, abort
from dotenv import dotenv_values
from launch import launch

app = Flask(__name__)

config = dotenv_values()
app.config.from_mapping(config)


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
    return jsonify({
        'ok': True,
    })


@app.route('/task/launch', methods=('POST',))
def launch_task():
    req = request.get_json()
    task = req.get('task')
    launch_options = req.get('launch')
    train_params = req.get('train')
    # grouped -> flatten
    for k in train_params:
        p = train_params[k]
        if isinstance(p, dict):
            train_params.pop(k)
            train_params.update(p)
    pid = launch(app.config, task, launch_options, train_params)

    return jsonify({
        'pid': pid,
    })


@app.route('/task/<id>', methods=('GET',))
def task_status(id):
    return jsonify({'id': id,
                    'status': 'ok',
                    })


def get():
    return app


if __name__ == '__main__':
    app.run()
