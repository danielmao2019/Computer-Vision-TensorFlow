from RCNN import RCNN

if __name__ == "__main__":
    model = RCNN()
    model = model.build(input_shape=(512, 512, 3))
    model.summary()
