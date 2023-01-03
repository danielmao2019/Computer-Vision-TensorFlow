class Sequential(object):

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, input_list, label_list):
        paired = list(zip(input_list, label_list))
        transformed = [self.transform(pair) for pair in paired]
        input_list, label_list = zip(*transformed)
        return input_list, label_list
