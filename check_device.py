import os
import torch
import accelerate

print(os.environ)

device_index=0

allocated = torch.cuda.memory_allocated(device_index)
reserved = torch.cuda.memory_reserved(device_index)
# stat = torch.cuda.memory_stats(device_index)
# print(stat)

k = 1024
g = k * k * k

d = accelerate.utils.get_max_memory()
dm = {i: (n / g) for i, n in d.items()}
print(f'max memory: {dm}')

free, total = torch.cuda.mem_get_info(device_index)
occupied = total - free
occupied_gb = occupied / g
print(f'occupied: {occupied_gb}')
