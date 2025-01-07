import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# Load the MNIST dataset from TensorFlow/Keras
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
x_test = x_test.astype("float32") / 255.0

# Create data directory if not exists
if not os.path.exists("data"):
    os.makedirs("data")

# Save random images and their labels to CSV and JPEG
for i in range(10):
    # Pick a random index from the test set
    idx = np.random.randint(0, x_test.shape[0])

    # Extract the image and label
    image = x_test[idx]  # shape: (28,28)
    label = y_test[idx]

    # Flatten the image into (784,) and then reshape to (784,1)
    image_flat = image.reshape(784, 1)

    # Save the single image to CSV with label in filename
    filename_csv = f"data/single_image_label_{label}_{i + 1}.csv"
    np.savetxt(filename_csv, image_flat, delimiter=",")

    # Save the image as JPEG with label in filename
    filename_jpeg = f"data/single_image_label_{label}_{i + 1}.jpeg"
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Hide axes
    plt.savefig(filename_jpeg, format='jpeg')

    # Close the plot to free up memory
    plt.close()

    print(f"Image {i + 1}:")
    print(f"  Selected test image index: {idx}")
    print(f"  True label: {label}")
    print(f"  CSV file created: {filename_csv}")
    print(f"  JPEG file created: {filename_jpeg}")
