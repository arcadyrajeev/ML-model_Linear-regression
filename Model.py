
import tensorflow as tf

"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential, load_model

"""

# creating dataset
X = tf.range(-100, 220, 4)

y = 2*X + 10

# setting up the data

X = tf.cast(X, dtype=tf.float32)
y = tf.cast(y, dtype=tf.float32)

X_train = X[:64]
y_train = y[:64]

X_test = X[64:]
y_test = y[64:]

print(X_train, y_train)

#Visualizing Data
'''plt.figure(figsize = (10,7))

plt.scatter(X_train, y_train, c= "b", label="Training Data")

plt.scatter(X_test, y_test, c= "g", label="Testing Data")

plt.legend();
plt.show()'''


# creating the model

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, input_shape=[1], activation="relu", name='input_layer0' ),
    tf.keras.layers.Dense(100, input_shape=[1], activation=None, name='input_layer1' ),
    tf.keras.layers.Dense(100, input_shape=[1], activation="relu", name='input_layer2' ),
    tf.keras.layers.Dense(1, name='output_layer')
], name= "Modelfirst")

# compile the model

model.compile(loss=tf.keras.losses.mae,
              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0039),
              metrics=["mae"])

#visualizing model
model.summary()

#plot_model(model=model, show_shapes=True)

# fit model

model.fit(X_train, y_train, epochs=100)



