
import numpy as np
from sklearn.preprocessing import normalize

def unit_norm(x):
    """
    x: a 2D array: (batch_size, vector_length)
    """
    return normalize(x, axis=1)
import functools
import operator



def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def reshape_2d(x):
    if len(x.shape) > 2:
        # Reshape to [#num_examples, ?]
        batch_size = x.shape[0]
        num_dim = functools.reduce(operator.mul, x.shape, 1)
        x = x.reshape((batch_size, num_dim/batch_size))
    return x


def get_normalizer_by_name(self, name):
    d = {'unit_norm': unit_norm, 'softmax': softmax, 'none': lambda x:x}
    return d[name]

def get_distance(layer_id, normalizer_name, metric_name, squeezers_name, X1, X2=None):

    normalize_func = get_normalizer_by_name(normalizer_name)
    input_to_normalized_output = lambda x: normalize_func(reshape_2d(eval_layer_output(x, layer_id)))

    val_orig_norm = input_to_normalized_output(X1)

    if X2 is None:
        vals_squeezed = []
        for squeezer_name in squeezers_name:
            squeeze_func = get_squeezer_by_name(squeezer_name)
            val_squeezed_norm = input_to_normalized_output(squeeze_func(X1))
            vals_squeezed.append(val_squeezed_norm)
        distance = calculate_distance_max(val_orig_norm, vals_squeezed, metric_name)
    else:
        val_1_norm = val_orig_norm
        val_2_norm = input_to_normalized_output(X2)
        distance_func = get_metric_by_name(metric_name)
        distance = distance_func(val_1_norm, val_2_norm)

    return distance

