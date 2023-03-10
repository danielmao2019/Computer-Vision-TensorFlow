class Dataloader(object):

    def __init__(self, dataset, shuffle=True, preprocessor=None, batch_size=None):
        """
        Args:
            dataset (data.datasets.Dataset).
            shuffle (bool).
            preprocessor (data.preprocess.Preprocessor).
            batch_size (int).
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        if shuffle:
            self.dataset.shuffle()
        # TODO: vectorize mapping
        if preprocessor:
            self.dataset.map(preprocessor)
        if batch_size:
            self.dataset.batch(batch_size)
        self.dataset.prefetch()

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for element in self.dataset:
            yield element
