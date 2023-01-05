"""
DATASETS API.
"""
from data.datasets.Dataset import Dataset
from data.datasets.MNIST.MNISTDataset import MNISTDataset
from data.datasets.OxfordIIITPets.OxfordIIITPetsDataset import OxfordIIITPetsDataset


__all__ = (
    "Dataset",
    "MNISTDataset",
    "OxfordIIITPetsDataset",
)
