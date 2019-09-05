import random
import pandas as pd


def test_train_split(dataset, train_size=1.0, seed=1234, shuffle=True):
    """
    Split test and train dataset
    :param dataset: the dataset
    :param train_size: the training percentange
    :param seed: the random seed
    :param shuffle: shuffle with respect to 'SessionId'
    :return: the training and test splits
    """
    if shuffle:
        groups = [df for _, df in dataset.groupby('SessionId')]
        random.Random(seed).shuffle(groups)
        dataset = pd.concat(groups).reset_index(drop=True)
    all_session_ids = list(dataset.groupby('SessionId', sort=False).groups.keys())

    m = len(all_session_ids)
    train_end = int(train_size * m)
    if train_end<m:
        sess = all_session_ids[train_end]
        split_index = dataset[dataset.SessionId == sess].iloc[0].name

        # split data according to the shuffled index and the holdout size
        train_split = dataset[:split_index]
        test_split = dataset[split_index:]
    else:
        train_split = dataset[:]
        test_split = dataset[:0]

    return train_split, test_split
