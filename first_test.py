import tensorflow as tf
import matplotlib.pyplot as plt #plotting library
import numpy as np

# In this test we use the Keras API (keras.io) who is a high-level neural networks API

# MNIST is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images
mnist = tf.keras.datasets.mnist

# Here we load MNIST datas. 
# train var are array of grayscale image data with shape 
# test var are array of digit labels (integers in range 0-9) with shape
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize datas to a scale of 1 instead of 255 for better understanding
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# define the model to Sequential.
# The Sequential model is a linear stack of layers. This the go-to standard model.
model = tf.keras.models.Sequential()

# This is the first input layer but because we don't want a 28x28 multi-dimensional array we flatten the input datas
model.add(tf.keras.layers.Flatten())

# Now we add 2 hidden regular densely-connected neuronal network layers
# The first parameter is the number of NEURONS in the layer
# we use relu as default activation function (standard)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# finally the output layer, in our case we have 10 choices sor that's why we need 10 neurons
# it's like a probability distribution
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Here we compile the model
# adam is the default go-to optimizer and same for sparse_categorical_crossentropy as loss
model.compile(optimizer='adam',
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])

# starts training
# Epoch is just a "full pass" through your entire training dataset. So if you just train on 1 epoch, then the neural network saw each unique sample once. 3 epochs means it passed over your data set 3 times.
model.fit(x_train, y_train, epochs=3)

# Generates output predictions for the input samples
# it return Numpy array(s) of predictions.
prediction = model.predict([x_test])

# argmax returns the indices of the maximum values along an axis
print("The number guessed is " + str(np.argmax(prediction[3])))

# show img to compare the result announced
plt.imshow(x_test[3])
plt.show()