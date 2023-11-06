import os
import signal
import subprocess


# temp
def sync_file(data_base_dir, filename, logger):
    checkpoints_dir = f'{data_base_dir}/sd-models/models/Stable-diffusion'
    src_file = f'{checkpoints_dir}/{filename}'

    if not os.path.exists(src_file):
        logger.error(f'file not exists: {src_file}')

    target_host = 'connect.westa.seetacloud.com'
    target_port = '42190'
    target_user = 'root'
    target_checkpoints_dir = '/root/autodl-tmp/sd/models/Stable-diffusion'

    def preexec_function():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    cmd = f"rsync -rv -e 'ssh -p {target_port}' {src_file} {target_user}@{target_host}:{target_checkpoints_dir}/"
    logger.info(cmd)
    p = subprocess.Popen(cmd,
                     preexec_fn=preexec_function,
                     shell=True)
    return p.pid
