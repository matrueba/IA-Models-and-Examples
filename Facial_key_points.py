#dataset font https://www.kaggle.com/c/facial-keypoints-detection/data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Input, Flatten
from keras.activations import relu, sigmoid
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import MSE

"""
Detect and plot facial key_points of human faces
"""


def plot_examples(n_examples, X_train, y_train, img_cols, img_rows):

    plt.figure(figsize=(12, 12))

    for i in range(n_examples):
        plt.subplot(n_examples, n_examples, i+1)
        rand = np.random.randint(len(X_train))
        img = X_train[rand].reshape(img_cols, img_rows)
        plt.imshow(img, cmap='gray')
        kp = y_train[rand]
        plt.scatter(kp[0::2] * img_cols, kp[1::2] * img_rows, marker='x')
    plt.show()


def plot_predict(n_examples, X_val, y_val, predictions, img_cols, img_rows):
    
    plt.figure(figsize=(12, 12))

    for i in range(n_examples):
        plt.subplot(n_examples, n_examples, i+1)
        rand = np.random.randint(len(X_val))
        img = X_val[rand].reshape(img_cols, img_rows)
        plt.imshow(img, cmap='gray')
        kp = y_val[rand]
        pred = predictions[rand]
        plt.scatter(kp[0::2] * img_cols, kp[1::2] * img_rows, marker='x')
        plt.scatter(pred[0::2] * img_cols, pred[1::2] * img_rows, marker='x')
    plt.show()


def conv_model(input_shape, n_labels):

    model = Sequential()
    model.add(Conv2D(64, input_shape=input_shape, kernel_size=(5, 5), padding="same", activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
    model.add(Conv2D(32, kernel_size=(5, 5), padding="same", activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(n_labels, activation='linear'))

    return model

#Load the files
training_file = pd.read_csv('Datasets/Faces/training.csv')
training_file['Image'] = training_file['Image'].apply(lambda Image: np.fromstring(Image, sep=' '))
training_file = training_file.dropna()

#From the dataset extract the images and the keypoints
train_images = np.vstack(training_file['Image'].values)
train_keypoints = training_file[training_file.columns.drop('Image')].values

img_cols = 96
img_rows = 96
img_channels = 1
keypoints = 30 #15 keypoints Ordered in x,y pairs

X = train_images.reshape(-1, img_cols, img_rows, 1)
y = train_keypoints

#Normalize
X = X/256 
y = y/96

print(X.shape, y.shape)

#Split in train and validation data  and plot some examples
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
plot_examples(3, X_train, y_train, img_cols, img_rows)

#Create and compile the model
input_shape =(img_cols, img_rows, 1)
model = conv_model(input_shape=input_shape, n_labels=keypoints)
model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#Define train callbacks
my_callbacks = [
    EarlyStopping(patience=5),
    ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
]

#Train the model and make the predictions
history = model.fit(x=X_train, y=y_train, batch_size=32, epochs=50, verbose=2,
 callbacks=my_callbacks, validation_data=(X_val, y_val))
predictions = model.predict(X_val)

#Plot the error
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Plot the loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Plot the predictions of some examples
plot_predict(5, X_val, y_val, predictions, img_cols, img_rows)