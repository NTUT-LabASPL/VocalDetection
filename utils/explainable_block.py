import numpy as np


def explainable_block(result, result_origin):
    """
    To calculate reversed probability.
        result == np.where(cls.predict(test_ds) >= 0.5, 1, 0)
        result_origin == np.where(cls_origin.predict(test_ds) >= 0.5, 1, 0)
    """
    reversed = 0
    result = np.array(result)
    result_origin = np.array(result_origin)
    for i in range(result.shape[0]):
        if result_origin[i][0] != result[i][0]:
            reversed += 1

    return reversed / result.shape[0]
