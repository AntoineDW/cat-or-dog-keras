# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    validation_split = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
    'data/train',
    target_size = (200, 200),
    batch_size = 64,
    color_mode = 'rgb',
    class_mode = 'binary',
    subset = 'training'
)

testing_set = train_datagen.flow_from_directory(
    'data/train',
    target_size = (200, 200),
    batch_size = 32,
    color_mode = 'rgb',
    class_mode = 'binary',
    subset = 'validation'
)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same', input_shape = (200, 200, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'))
model.add(Dense(1, activation = 'sigmoid'))

opt = SGD(lr = 0.001, momentum = 0.9)

model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit_generator(
    training_set,
    steps_per_epoch = len(training_set),
    epochs = 40,
    validation_data = testing_set,
    validation_steps = len(testing_set)
)

model_json = model.to_json()

with open("model_cnn.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model_cnn.h5")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()