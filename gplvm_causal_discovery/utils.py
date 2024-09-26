import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess(x, y, test_size=0.2, shuffle_seed=0):
    '''
    Normalise and shuffle data, then split into train and test sets

    Args:
        data: data to be split (x, y) shape (2, n, 1)
        test_size: proportion of data to use for testing

    Returns:
        trainval_data_norm: normalised train and validation data (x, y)
        test_data_norm: normalised test data (x, y)
    '''

    #random generator
    rng = np.random.default_rng(seed=shuffle_seed)

    #shuffle data
    # idx = np.random.permutation(x.shape[0])
    idx = rng.permutation(x.shape[0])
    data = np.concatenate((x, y), axis=1)[idx]

    if test_size == 0:
        scaler_train = StandardScaler()
        train_data_norm = scaler_train.fit_transform(data)
        x_train_norm = train_data_norm.T[0].reshape(-1, 1)
        y_train_norm = train_data_norm.T[1].reshape(-1, 1)
        return (x_train_norm, y_train_norm), None
    
    else:
        #split data into train and test
        test_n = int(test_size * x.shape[0])
        data_trainval = data[:-test_n]
        data_test = data[-test_n:]

        #normalise data: fit scaler to trainval data, then transform trainval and test data
        scaler_trainval = StandardScaler()
        trainval_data_norm = scaler_trainval.fit_transform(data_trainval)
        test_data_norm = scaler_trainval.transform(data_test)

        #split into x and y 
        x_trainval_norm = trainval_data_norm.T[0].reshape(-1, 1)
        y_trainval_norm = trainval_data_norm.T[1].reshape(-1, 1)
        x_test_norm = test_data_norm.T[0].reshape(-1, 1)
        y_test_norm = test_data_norm.T[1].reshape(-1, 1)

        return (x_trainval_norm, y_trainval_norm), (x_test_norm, y_test_norm)