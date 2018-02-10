





def get_dataset(name,
                batch_size = 100,
                cycle = True,
                shuffle = True,
                num_batches = -1):

    """Get a dataset

    Args:
        name: "notMNIST" or "FaceScrub"
        batch_size: Mini-batch size
        cycle: Whether to stop after one full epoch.
        shuffle: Whether to shuffle the data.
        num_batches:



    Returns:
        Dataset Iterator
    """

    if name == 'notMNIST':
        print('a')
    elif name == 'FaceScrub':
        print('b')