class Dataloader(object):

    def __init__(self, dataset, shuffle, preprocessor):
        """
        Args:
            dataset (data.datasets.Dataset).
            shuffle (bool).
            preprocessor (data.preprocess.Preprocessor).
        """
        self.dataset = dataset
        if shuffle:
            self.dataset.shuffle()
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for input, label in self.dataset:
            yield self.preprocessor(input, label)
