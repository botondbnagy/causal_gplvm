from data.an_pairs.generate_an_pairs import ANPairs
from data.tueb.generate_tueb import TubingenPairs
from data.cha.generate_cha import ChaPairs
import numpy as np



def get_an_pairs_dataset(data_path, pair_index):
    data_gen = ANPairs(path=data_path)

    x, y, weight, target = data_gen.return_pair(pair_index)
    return x, y, weight, target

def get_tueb_dataset(data_path, pair_index):
    data_gen = TubingenPairs(path=data_path)

    x, y, weight, target, new_pair_index = data_gen.return_single_1D_set(pair_index)
    return x, y, weight, target, new_pair_index

def get_cha_dataset(data_path, pair_index):
    data_gen = ChaPairs(path=data_path)
    i = pair_index - 1

    x_all, y_all, weight_all, target_all = data_gen.return_pairs()
    return x_all[i], y_all[i], weight_all[i], target_all[i]

def get_inf_data(func, noise_std, num_points):
    #draw x from normal distribution with mean = 0, std = 1
    left_n = num_points // 2
    x = np.random.normal(-1, 1, left_n)
    x = np.concatenate((x, np.random.normal(1, 0.5, num_points - left_n)))
    #draw x from uniform distribution between -1 and 1
    #x = np.random.uniform(-10, 10, num_points)
    #draw noise from normal distribution with std = noise_std
    noise = np.random.normal(0, noise_std, num_points)
    #calculate y
    if func == 'cubic':
        f = lambda x: x**3

    elif func == 'sinc':
        f = lambda x: np.sinc(x * 1)
        
    y = f(x) + noise
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    target = 1
    return x, y, target
