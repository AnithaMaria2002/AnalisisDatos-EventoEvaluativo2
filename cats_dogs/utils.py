import numpy as np # linear algebra
import pandas as pd # CSV file
from sklearn import preprocessing
import scipy.io.wavfile as sci_wav  # Open wav files

ROOT_DIR = '../input/cats_dogs/'


def read_wav_files(wav_files):
    '''Returns a list of audio waves
    Params:
        wav_files: List of .wav paths
    
    Returns:
        List of audio signals
    '''
    if not isinstance(wav_files, list):
        wav_files = [wav_files]
    return [sci_wav.read(ROOT_DIR + f)[1] for f in wav_files]


def get_trunk(_X, idx, sample_len, rand_offset=False):
    '''Returns a trunk of the 1D array <_X>

    Params:
        _X: the concatenated audio samples
        idx: _X will be split in <sample_len> items. _X[idx]
        rand_offset: boolean to say whether or not we use an offset
    '''
    randint = np.random.randint(10000) if rand_offset is True else 0
    start_idx = (idx * sample_len + randint) % len(_X)
    end_idx = ((idx+1) * sample_len + randint) % len(_X)
    if end_idx > start_idx:  # normal case
        return _X[start_idx : end_idx]
    else:
        return np.concatenate((_X[start_idx:], _X[:end_idx]))


def get_augmented_trunk(_X, idx, sample_len, added_samples=0):
    X = get_trunk(_X, idx, sample_len)

    # Add other audio of the same class to this sample
    for _ in range(added_samples):
        ridx = np.random.randint(len(_X))  # random index
        X = X + get_trunk(_X, ridx, sample_len)

    # One might add more processing (like adding noise)

    return X


def dataset_gen(is_train=True, batch_shape=(20, 16000)):
    '''This generator is going to return training batchs of size <batch_shape>
    
    Params:
        batch_shape: a tupple (or list) consisting of 2 arguments, the number of 
            samples per batchs and the number datapoints per samples (16000 = 1s)
    '''
    n_samples = batch_shape[0]
    sample_len = batch_shape[1]

    X_cat = dataset['X_train_cat'] if is_train else dataset['X_test_cat']
    X_dog = dataset['X_train_dog'] if is_train else dataset['X_test_dog']

    # Random permutations (for X indexes)
    nbatch = int(max(len(X_cat), len(X_cat)) / sample_len)
    cat_p = np.random.permutation(nbatch)  # nCat > nDog => Dog will be out of range
    dog_p = np.random.permutation(nbatch)  # use a % for them

    # Loop through the <bidx> batchs
    for bidx in range(nbatch):
        y_batch = np.zeros(n_samples)
        X_batch = np.zeros(batch_shape)

        # Loop through the <sidx> batch_samples
        for sidx in range(n_samples):
            y_batch[sidx] = sidx % 2
            _X = X_cat if sidx == 0 else X_dog
            if is_train:
                X_batch[sidx] = get_augmented_trunk(_X,
                                                    idx=sidx,
                                                    sample_len=sample_len,
                                                    added_samples=2)
            else:
                X_batch[sidx] = get_trunk(_X, sidx, sample_len)
        
        yield (X_batch.reshape(n_samples, sample_len, 1),
               y_batch.reshape(-1, 1) )


df = pd.read_csv('../input/train_test_split.csv')

dataset = {}
for k in ['train_cat', 'train_dog', 'test_cat', 'test_dog']:
    v = list(df[k].dropna())
    v = read_wav_files(v)
    v = np.concatenate(v).astype('float32')
    print (k)

    # Compute mean and variance
    if k == 'train_cat':
        dog_std = dog_mean = 0
        cat_std, cat_mean = v.std(), v.mean()
    elif k == 'train_dog':
        dog_std, dog_mean = v.std(), v.mean()

    # Mean and variance suppression
    std, mean =  (cat_std, cat_mean) if 'cat' in k else (dog_std, dog_mean)
    v = v / std - mean

    dataset[k] = v