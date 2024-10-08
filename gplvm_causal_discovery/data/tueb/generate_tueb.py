import os
from typing import Tuple, Generator
import numpy as np
from tqdm import tqdm


class TubingenPairs:
    def __init__(self, path='./files'):
        self.data = dict()
        self.tueb_1d_ids = []

        with open(os.path.join(path, 'pairmeta.txt'), 'r') as f:
            for line_raw in f:
                line = line_raw.split(' ')
                data_id = int(line[0])
                cause_rng = np.arange(int(line[1]) - 1, int(line[2]))
                effect_rng = np.arange(int(line[3]) - 1, int(line[4]))
                wt = float(line[5])
                self.data[data_id] = {
                    'cause_inds': cause_rng,
                    'effect_inds': effect_rng,
                    'weight': wt,
                    'target': 1 if cause_rng[0] == 0 else -1 #if column 0 is cause, target is 1
                }

        files = os.listdir(path)
        files = [f for f in files
                 if f.startswith('pair0') and '_des' not in f]
        for file in tqdm(files, desc='Load cause-effect pairs',
                         total=len(files)):
            file_id = int(file.lstrip('pair').rstrip('.txt'))
            data = np.loadtxt(os.path.join(path, file))
            cause = data[:, self.data[file_id]['cause_inds']]
            effect = data[:, self.data[file_id]['effect_inds']]
            self.data[file_id]['cause'] = cause
            self.data[file_id]['effect'] = effect
            
            #store list of 1 dimensional datasets
            if cause.shape[1] == 1 and effect.shape[1] == 1:
                self.tueb_1d_ids.append(file_id)

        self.tueb_1d_ids = np.sort(self.tueb_1d_ids)
        #print(f'1D datasets: {self.tueb_1d_ids}')

    def return_single_set(self, set_id) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a single pair dataset from the full set of pairs
        :param set_id: Integer ID of the dataset
        :return: Tuple of (cause, effect), with data in 2D Numpy array
        """
        try:
            set_id = int(set_id)
            dataset = self.data[set_id]
        except ValueError:
            raise(f'Dataset key {set_id} is not valid - '
                  f'please enter an integer-like value.')
        except KeyError:
            raise(f'Dataset key {set_id} is not present in the data.')
        return dataset['cause'], dataset['effect'], dataset['weight'], dataset['target']
    
    def return_single_1D_set(self, set_id) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a single pair dataset from the full set of pairs,
        excluding high dimensional datasets
        :param set_id: Integer ID of the dataset (core index + 1)
        :return: Tuple of (cause, effect), with data in 2D Numpy array
        """
        try:
            set_id = int(set_id)
            #get 1 dimensional x and y dataset indices only
            remapped_set_id = self.tueb_1d_ids[set_id - 1]
            dataset = self.data[remapped_set_id]
        except ValueError:
            raise(f'Dataset key {set_id} is not valid - '
                  f'please enter an integer-like value.')
        except KeyError:
            raise(f'Dataset key {set_id} is not present in the data.')
        return dataset['cause'], dataset['effect'], dataset['weight'], dataset['target'], remapped_set_id
        
    



    def pairs_generator(self) -> Generator[
        Tuple[np.ndarray, np.ndarray, float], None, None
    ]:
        """
        Produce a generator object that will yield each of the cause-effect
        pair datsets.

        Weight factor is also returned to weight the significance of the pair
        within the whole dataset (used to account for effectively duplicate
        data across dataset pairs).
        :return: Tuple of (cause, effect, weight) - cause, effect are 2D numpy
        arrays
        """
        for dataset in self.data.values():
            yield dataset['cause'], dataset['effect'], dataset['weight'], dataset['target']

    def return_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return all cause-effect pairs as a single dataset
        :return: Tuple of (cause, effect), with data in 2D Numpy array
        """
        cause = np.concatenate([dataset['cause'] for dataset in self.data.values()], axis=0)
        effect = np.concatenate([dataset['effect'] for dataset in self.data.values()], axis=0)
        weights = np.concatenate([dataset['weight'] for dataset in self.data.values()], axis=0)
        targets = np.concatenate([dataset['target'] for dataset in self.data.values()], axis=0)
        return cause, effect, weights, targets
