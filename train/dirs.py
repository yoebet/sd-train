def get_train_dir(TRAIN_DATA_DIR, task_id, sub_dir=None):
    if sub_dir == '_':
        sub_dir = None
    trains_dir = f'{TRAIN_DATA_DIR}/trains'
    if sub_dir is not None:
        train_dir = f'{trains_dir}/{sub_dir}/t_{task_id}'
    else:
        train_dir = f'{trains_dir}/t_{task_id}'

    return train_dir


def get_logging_dir(TRAIN_DATA_DIR, sub_dir=None):
    logging_dir = f'{TRAIN_DATA_DIR}/logs/hot'
    if sub_dir is not None and sub_dir != '_':
        return f'{logging_dir}/{sub_dir}'
    return logging_dir
