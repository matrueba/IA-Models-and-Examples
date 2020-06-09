#Font of data https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip

import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

def plot_examples(n_examples, train_cats_dir, train_dogs_dir):

    plt.figure(figsize=(15, 15))

    for i in range(n_examples):
        plt.subplot(n_examples, n_examples, i+1)
        rand_cat = np.random.choice(os.listdir(train_cats_dir))
        rand_dog = np.random.choice(os.listdir(train_dogs_dir))
        if i%2 == 0:
            img = mpimg.imread(os.path.join(train_cats_dir, rand_cat))
        else:
            img = mpimg.imread(os.path.join(train_dogs_dir, rand_dog))
        plt.imshow(img)
    plt.show()

#Load the data
local_zip = 'Datasets/CatsAndDogs/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('Datasets/CatsAndDogs/tmp')
zip_ref.close()

base_dir = 'Datasets/CatsAndDogs/tmp/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

print ('total training cat images:', len(os.listdir(train_cats_dir)))
print ('total training dog images:', len(os.listdir(train_dogs_dir)))
print ('total validation cat images:', len(os.listdir(validation_cats_dir)))
print ('total validation dog images:', len(os.listdir(validation_dogs_dir)))

plot_examples(8, train_cats_dir, train_dogs_dir)

#Crete the convolutional model
model = Sequential()
model.add(Conv2D(64, 3, activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPool2D(2))
model.add(Conv2D(32, 3, activation='relu'))
model.add(MaxPool2D(2))
model.add(Conv2D(16, 3, activation='relu'))
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])


#Train the model using generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        train_dir, 
        target_size=(150, 150), 
        batch_size=20,        
        class_mode='binary')


validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')


history = model.fit_generator(
      train_generator,
      steps_per_epoch=100, 
      epochs=20,
      validation_data=validation_generator,
      validation_steps=50,  
      verbose=2)

#Plot results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.figure()
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
plt.show()

model.save('Datasets/CatsAndDogs/model.h5')
os.remove('Datasets/CatsAndDogs/tmp')