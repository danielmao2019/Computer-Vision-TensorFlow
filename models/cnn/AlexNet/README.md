# models.cnn.AlexNet

## Experimental Notes

* [Tue Jan 03, 2023] Training AlexNet on MNIST dataset with input resolution 128x128 diverges with SGD optimizer, learning rate 0.01, other arguments default. Observed loss decrease after setting learning rate to 1.0e-03.
