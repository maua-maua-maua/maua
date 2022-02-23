import os
from copy import deepcopy
from functools import wraps

import joblib


def hash(tensor_array_int_obj):
    if isinstance(tensor_array_int_obj, (np.ndarray, torch.Tensor)):
        if isinstance(tensor_array_int_obj, torch.Tensor):
            array = tensor_array_int_obj.detach().cpu().numpy()
        else:
            array = tensor_array_int_obj
        array = deepcopy(array)
        byte_tensor = (normalize(array) * 255).ravel().astype(np.uint8)
        hash = 0
        for ch in byte_tensor[:1024:4]:
            hash = (hash * 281 ^ ch * 997) & 0xFFFFFFFF
        return str(hex(hash)[2:].upper().zfill(8))
    if isinstance(tensor_array_int_obj, (float, int, str, bool)):
        return str(tensor_array_int_obj)
    return ""


def cache_to_workspace(name):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            arghash = "_".join([hash(a) for a in [*args, *kwargs.values()]])
            cache_file = f"workspace/audio_cache/{name}_{arghash}.npy"
            if not os.path.exists(cache_file):
                result = function(*args, **kwargs)
                joblib.dump(result, cache_file, compress=9)
            else:
                result = joblib.load(cache_file)
            return result

        return wrapper

    return decorator


from .audio import *
from .features import *
from .inputs import *
from .postprocess import *
from .util import *
