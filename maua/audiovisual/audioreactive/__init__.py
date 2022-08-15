import os
from functools import wraps

import joblib

from ...ops.io import hash


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
                try:
                    result = joblib.load(cache_file)
                except:
                    result = function(*args, **kwargs)
                    joblib.dump(result, cache_file, compress=9)
            return result

        return wrapper

    return decorator


from .audio import *
from .latent import *
from .mir import *
from .noise import *
from .signal import *
from .util import *
