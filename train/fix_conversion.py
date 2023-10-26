import sys

from safetensors import safe_open
from safetensors.torch import save_file


# fix convert_diffusers_to_original_stable_diffusion
# https://github.com/huggingface/diffusers/issues/4790

def fix_diffusers_model_conversion(load_path: str, save_path: str = None):
    if save_path is None:
        save_path = load_path

    # load original
    tensors = {}
    with safe_open(load_path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # migrate
    new_tensors = {}
    for k, v in tensors.items():
        new_key = k
        # only fix the vae
        if 'first_stage_model.' in k:
            # migrate q, k, v keys
            new_key = new_key.replace('.to_q.weight', '.q.weight')
            new_key = new_key.replace('.to_q.bias', '.q.bias')
            new_key = new_key.replace('.to_k.weight', '.k.weight')
            new_key = new_key.replace('.to_k.bias', '.k.bias')
            new_key = new_key.replace('.to_v.weight', '.v.weight')
            new_key = new_key.replace('.to_v.bias', '.v.bias')
        new_tensors[new_key] = v

    # save
    save_file(new_tensors, save_path)


if __name__ == '__main__':
    file = sys.argv[1]
    fix_diffusers_model_conversion(file)
