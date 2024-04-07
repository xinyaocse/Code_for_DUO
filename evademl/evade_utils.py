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
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def reshape_2d(x):
    if len(x.shape) > 2:
        # Reshape to [#num_examples, ?]
        batch_size = x.shape[0]
        num_dim = functools.reduce(operator.mul, x.shape, 1)
        x = x.reshape((batch_size, num_dim / batch_size))
    return x


def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False


def parse_params(params_str):
    params = []

    for param in params_str.split('_'):
        param = param.strip()
        if param.isdigit():
            param = int(param)
        elif isfloat(param):
            param = float(param)
        else:
            continue
        params.append(param)

    return params


from evademl.squeeze import *


def get_squeezer_by_name(name, func_type='python'):
    squeezer_list = ['none',
                     'bit_depth_random',
                     'bit_depth',
                     'binary_filter',
                     'binary_random_filter',
                     'adaptive_binarize',
                     'otsu_binarize',
                     'median_filter',
                     'median_random_filter',
                     'median_random_size_filter',
                     'non_local_means_bw',
                     'non_local_means_color',
                     'adaptive_bilateral_filter',
                     'bilateral_filter',
                     'magnet_mnist',
                     'magnet_cifar10',
                     'denoising'
                     ]

    for squeezer_name in squeezer_list:
        if name.split('_')[0] == squeezer_name.split('_')[0]:
            func_name = "%s_py" % squeezer_name if func_type == 'python' else "%s_tf" % squeezer_name
            params_str = name[len(squeezer_name):]

            # Return a list
            args = parse_params(params_str)
            # print ("params_str: %s, args: %s" % (params_str, args))

            return lambda x: globals()[func_name](*([x.detach().cpu().numpy()] + args))

    raise Exception('Unknown squeezer name: %s' % name)


from scipy.stats import entropy


def kl(x1, x2):
    assert x1.shape == x2.shape
    # x1_2d, x2_2d = reshape_2d(x1), reshape_2d(x2)

    # Transpose to [?, #num_examples]
    x1_2d_t = x1.transpose()
    x2_2d_t = x2.transpose()

    # pdb.set_trace()
    e = entropy(x1_2d_t, x2_2d_t)
    e[np.where(e == np.inf)] = 2
    return e


l1_dist = lambda x1, x2: np.sum(np.abs(x1 - x2), axis=tuple(range(len(x1.shape))[1:]))
l2_dist = lambda x1, x2: np.sum((x1 - x2) ** 2, axis=tuple(range(len(x1.shape))[1:])) ** .5


def get_metric_by_name(name):
    d = {'kl_f': lambda x1, x2: kl(x1, x2), 'kl_b': lambda x1, x2: kl(x2, x1), 'l1': l1_dist, 'l2': l2_dist}
    return d[name]


def get_normalizer_by_name(name):
    d = {'unit_norm': unit_norm, 'softmax': softmax, 'none': lambda x: x}
    return d[name]


def calculate_distance_max(val_orig, vals_squeezed, metric_name):
    distance_func = get_metric_by_name(metric_name)

    dist_array = []
    for val_squeezed in vals_squeezed:
        dist = distance_func(val_orig, val_squeezed)
        dist_array.append(dist)

    dist_array = np.array(dist_array)
    return np.max(dist_array, axis=0)


def video_squeeze_func(squeeze_func, input):
    output_array = torch.zeros_like(input)
    for i in range(16):
        output_frame = squeeze_func(input[:, :, i].permute(0, 2, 3, 1))
        output_temp = torch.tensor(output_frame).permute(0, 3, 1, 2)
        output_array[:, :, i] = output_temp

    return output_array


def get_distance(X1, X2=None, model=None):
    # normalizer_name, metric_name, squeezers_name = 'unit_norm', 'l1', 'median_smoothing_2,bit_depth_4'
    normalizer_name, metric_name, squeezers_name = 'unit_norm', 'l1', 'denoising'

    squeezers_name = squeezers_name.split(',')
    normalize_func = get_normalizer_by_name(normalizer_name)
    input_to_normalized_output = lambda x: normalize_func(model(x).detach().cpu().numpy())
    val_orig_norm = input_to_normalized_output(X1)

    if X2 is None:
        vals_squeezed = []
        for squeezer_name in squeezers_name:
            squeeze_func = get_squeezer_by_name(squeezer_name)
            squeeze_V1 = video_squeeze_func(squeeze_func, X1)
            val_squeezed_norm = input_to_normalized_output(squeeze_V1)
            vals_squeezed.append(val_squeezed_norm)
        distance = calculate_distance_max(val_orig_norm, vals_squeezed, metric_name)
    else:
        val_1_norm = val_orig_norm
        val_2_norm = input_to_normalized_output(X2)
        distance_func = get_metric_by_name(metric_name)
        distance = distance_func(val_1_norm, val_2_norm)
    return distance, vals_squeezed
