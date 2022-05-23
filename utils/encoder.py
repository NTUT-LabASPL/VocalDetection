import json
import numpy as np
import multiprocessing.managers


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, multiprocessing.managers.ListProxy):
            return np.array(obj).tolist()
        return json.JSONEncoder.default(self, obj)
