import numpy as np
from mnist import MNIST

mndata = MNIST('path/to/mnist')
images, labels = mndata.load_training()

X = np.array(images).T / 255.0  # normalize to [0,1]

Y = np.zeros((10, len(labels)))
for i, lab in enumerate(labels):
    Y[lab, i] = 1

np.savetxt('data/train_data.csv', X, delimiter=',')
np.savetxt('data/train_labels.csv', Y, delimiter=',')
