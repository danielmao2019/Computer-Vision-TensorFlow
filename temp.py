from models.cnn.SENet import SENet
from models.cnn.MobileNets.MobileNetV3 import MobileNetV3

if __name__ == "__main__":
    model = MobileNetV3(model_type='small', output_dim=10)
    model = model.build()
    model.summary()
