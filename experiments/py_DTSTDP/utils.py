import numpy as np
import copy
import os
import pathlib


def isfloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def try_parse_array_spec(arg):
    if type(arg) == str:
        args = arg.split(':')
        if len(args) == 3:
            if all([isfloat(a) for a in args]):
                start = float(args[0])
                increment = float(args[1])
                end = float(args[2])
                return list(np.around(np.arange(start, end + increment, increment), 4))
            else:
                return arg
        else:
            return arg
    else:
        return arg


def parse_config_dict(dictionary):
    for key, val in dictionary.items():
        if type(val) == dict:
            parse_config_dict(val)
        else:
            dictionary[key] = try_parse_array_spec(dictionary[key])


def dict_prod(key, vals, dict_list):
    dict_list_prod = []
    for val in vals:
        dict_list_copy = copy.deepcopy(dict_list)
        for dictionary in dict_list_copy:
            dictionary[key] = val
            dict_list_prod.append(dictionary)
    return dict_list_prod


def dict_factor(dictionary):
    dict_list = [copy.copy(dictionary)]
    for key in dictionary:
        vals = dictionary[key]
        dict_list = dict_prod(key, vals, dict_list)
    return dict_list


def get_subfolders(file_name):
    file_name = file_name.replace('.py', '')
    file_name = file_name.split('/')
    file_name = file_name[-1]
    # print(file_name)
    project_folder = pathlib.Path(__file__).parent.parent.parent.absolute()
    # print(project_folder)
    data_folder = os.path.join(project_folder, 'data', file_name)
    # print(data_folder)
    figure_folder = os.path.join(project_folder, 'figures', file_name)
    # print(figure_folder)
    try:
        os.mkdir(data_folder)
    except:
        pass
        # raise Exception('failed to make {}'.format(data_folder))
    try:
        os.mkdir(figure_folder)
    except:
        pass
        # raise Exception('failed to make {}'.format(figure_folder))
    return project_folder, data_folder, figure_folder


def get_minmax_array(array, lag):
    min_array = np.zeros(len(array)-lag)
    max_array = np.zeros(len(array)-lag)
    flux_array = np.zeros(len(array)-lag)
    val_array = np.zeros(len(array)-lag)
    for i in range(len(min_array)):
        val, min, max, sign = get_min_max_in_range(array, i, lag)
        val_array[i] = val
        min_array[i] = min
        max_array[i] = max
        flux_array[i] = sign * (max - min)
    return val_array, min_array, max_array, flux_array


def get_min_max_in_range(array, index, lag):
    sub_array = array[index:(index+lag)]
    val = array[index]
    min = np.min(sub_array)
    max = np.max(sub_array)
    min_i = np.argmin(sub_array)
    max_i = np.argmax(sub_array)
    sign = np.sign(max_i - min_i)
    return val, min, max, sign


def get_lag_diff(array, lag):
    lag_diff = array[lag:] - array[:-lag]
    mean_lag_diff = np.mean(lag_diff)
    std_lag_diff = np.std(lag_diff)
    return mean_lag_diff, std_lag_diff


def orient_array(array1, array2):
    offset_error = np.zeros_like(array2)
    for i in range(len(array2)):
        offset_error[i] = np.sum((array1 - np.roll(array2, i))**2)
    return np.roll(array2, np.argmin(offset_error))


def orient_img(a):
    array = copy.copy(a)
    for i in range(1, array.shape[1]):
        array[:, i] = orient_array(array[:, i-1], array[:, i])
    return array