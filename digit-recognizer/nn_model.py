import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data = pd.read_csv('data/train.csv')

labels = data.pop('label')

def vectorize(n):
    """ Convert a number to a one-hot classification vector """
    arr = np.zeros((10))
    arr[n] = 1
    return arr

x = data.values
labels = labels.values

x_train, x_test, y_train, y_test = train_test_split(x, labels)

def compile_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(784),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model

model = compile_model()
history = model.fit(x_train, y_train, epochs=25, verbose=0)

hist = pd.DataFrame(history.history)

plt.plot(hist.loss[2:], label='loss')
plt.plot(hist.accuracy, label='accuracy')
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid()

plt.savefig("training.png")

## TESTING
y_pred = model.predict(x_test)
y_pred_labels = y_pred.argmax(axis=1)

model.evaluate(x_test, y_test)

confusion = confusion_matrix(y_test, y_pred_labels)
plt.matshow(confusion)

plt.xticks(np.arange(10))
plt.yticks(np.arange(10))

plt.xlabel("Prediction")
plt.ylabel("Ground truth")

plt.savefig("confusion.png")

## PREDICTION
x_test = pd.read_csv("data/test.csv").values

y_pred = model.predict(x_test)
y_pred_labels = y_pred.argmax(axis=1)

results = pd.DataFrame({'ImageId':np.arange(len(y_pred)), 'Label':y_pred_labels})
results.to_csv('predictions_nn.csv', index=False)
