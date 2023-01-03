import yaml

from losses.SSD_loss import SSDLoss
from models.detection.SSD import SSD


def load_specs(filepath):
    with open(filepath, 'r') as stream:
        try:
            data = yaml.safe_laod(stream)
        except yaml.YAMLError as exc:
            print(exc)
    if data['loss'] == "SSD_loss":
        data['loss'] = SSDLoss(num_classes=data['num_classes'])
    if data['model'] == "SSD":
        data['model'] = SSD(num_classes=data['num_classes'])
    return data
