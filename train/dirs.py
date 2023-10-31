def get_train_dir(data_base_dir, task_id, sub_dir=None):
    if sub_dir == '_':
        sub_dir = None
    trains_dir = f'{data_base_dir}/trains'
    if sub_dir is not None:
        train_dir = f'{trains_dir}/{sub_dir}/t_{task_id}'
    else:
        train_dir = f'{trains_dir}/t_{task_id}'

    return train_dir


def get_logging_dir(data_base_dir, sub_dir=None):
    logging_dir = f'{data_base_dir}/logs/hot'
    if sub_dir is not None and sub_dir != '_':
        return f'{logging_dir}/{sub_dir}'
    return logging_dir
