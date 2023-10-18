import logging
from flask import Flask, jsonify, request, Response, abort
from dotenv import dotenv_values

app = Flask(__name__)

config = dotenv_values()
app.config.from_mapping(config)

print(app.config)


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


def _response_text(sel):
    return Response('', mimetype='text/plain')


@app.route('/tasks', methods=('POST',))
def summit_task():
    return jsonify(('ok',))


@app.route('/tasks/<id>', methods=('GET',))
def task_status(id):
    return jsonify({'id': id,
                    'status': 'ok',
                    })


def get():
    return app


if __name__ == '__main__':
    app.run()
