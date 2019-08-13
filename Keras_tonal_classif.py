import numpy as np
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.datasets import imdb
np.load.__defaults__= (None, True, True, 'ASCII')

(training_X, training_y), (testing_X, testing_y) = imdb.load_data(num_words=10000)
data = np.concatenate((training_X, testing_X), axis=0)
targets = np.concatenate((training_y, testing_y), axis=0)

def vectorize(sequences, dimension = 10000):
  res = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    res[i, sequence] = 1
  return res
 
data = vectorize(data)
targets = np.array(targets).astype("float32")

X_test = data[:10000]
y_test = targets[:10000]
X_train = data[10000:]
y_train = targets[10000:]
model = models.Sequential()

model.add(layers.Dense(50, activation = "relu", input_shape=(10000,)))
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()

model.compile(
 optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"]
)
results = model.fit(
 X_train, y_train,
 epochs= 3,
 batch_size = 400,
 validation_data = (X_test, y_test)
)
print("Test-Accuracy:", np.mean(results.history["val_acc"]))