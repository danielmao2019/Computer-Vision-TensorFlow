class Preprocessor(object):

    def __init__(self, transforms):
        """
        Arguments:
            transforms (list): A list of transforms to be applied in order.
        """
        if not isinstance(transforms, list):
            raise TypeError(f"[ERROR] Argument \'transforms\' should be a list. Got {type(transforms)}.")
        self.transforms = transforms

    def __call__(self, input, label):
        for transform in self.transforms:
            input, label = transform(input, label)
        return input, label
