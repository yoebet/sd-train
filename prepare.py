import os
import time
import requests
from urllib.parse import urlparse
import hashlib
import shutil
import logging
import re

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def prepare_instance_images(config, task, skip_if_exists=False):
    data_base_dir = config['DATA_BASE_DIR']
    task_id = task.get('task_id', None)
    if task_id is None:
        task_id = str(int(time.time()))
    instance_images = task.get('instance_images', None)
    if instance_images is None:
        logging.warning(f'prepare (task_id={task_id}): no instance_images')
        return {'task_id': task_id}

    train_dir = f'{data_base_dir}/trains/t_{task_id}'
    instance_data_dir = f'{train_dir}/instance_images'

    if os.path.exists(instance_data_dir):
        if skip_if_exists:
            files = os.listdir(instance_data_dir)
            files = [f for f in files if re.match(r'\d+-', f)]
            if len(files) == len(instance_images):
                logging.info(f'instance_images exists, skip')
                return {'task_id': task_id}

        shutil.rmtree(instance_data_dir)
    os.makedirs(instance_data_dir, exist_ok=True)
    for i, url in enumerate(instance_images):
        if not url.startswith('http'):
            logging.warning(f'image must be url: {url}')
            continue
        image_path = urlparse(url).path
        image_name = os.path.basename(image_path)
        if '.' in image_name:
            ext = image_name.split('.')[-1]
            if ext not in ('jpg', 'jpeg', 'png'):
                logging.warning(f'image format: {ext} ({url})')
        else:
            ext = 'jpg'
        res = requests.get(url)
        if res.status_code == 200:
            image_hash = hashlib.sha1(res.content).hexdigest()
            # image_hash = '2466'
            file_name = f'{i + 1}-{image_hash}.{ext}'
            with open(f'{instance_data_dir}/{file_name}', 'wb') as f:
                # shutil.copyfileobj(res.content, f)
                f.write(res.content)
        else:
            logging.warning(f'download failed: {url}')

    return {'task_id': task_id}


if __name__ == '__main__':
    config = {
        'DATA_BASE_DIR': 'data'
    }
    task = {
        'task_id': '134',
        'instance_images': [
            'https://oss-prod.tishi.top/draw/sd/models/majicmixRealistic_v6-43331.jpg',
            'https://oss-prod.tishi.top/draw/sd/models/manmaruMix_v20-86277.jpg',
            'https://oss-prod.tishi.top/draw/sd/lora-models/33208-7934.jpg'
        ],
    }
    prepare_instance_images(config, task)
