import tensorflow as tf
from tensorflow.python.client import device_lib

from pynvml.smi import nvidia_smi


nvsmi = nvidia_smi.getInstance()


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_allocated_memory() -> int:
    # This is only a rough estimate because nvidia smi isn't very specific
    mem_sum = 0
    for usage_dict in nvsmi.DeviceQuery('memory.used')['gpu']:
        usage = usage_dict['fb_memory_usage']
        mem_sum += usage_to_bytes(usage)
        
    return mem_sum


def usage_to_bytes(usage) -> int:
    unit = usage['unit']
    used = usage['used']

    if 'K' in unit:
        return used * 1_000
    elif 'M' in unit:
        return used * 1_000_000
    elif 'G' in unit:
        return used * 1_000_000_000


def memory_usage_info(_print=False):
    mem_sum = 0
    for gpu in get_available_gpus():
        mem_bytes = tf.config.experimental.get_memory_info(gpu)['current']
        mem_sum += mem_bytes
        
    allocated = get_allocated_memory()
    usage = {'allocated': allocated, 'used': mem_sum}
        
    if _print:
        print(usage)
    else:
        return usage
