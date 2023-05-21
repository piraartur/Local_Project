import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, RMSprop

# Get the absolute paths of the dataset and model files
dataset_path = os.path.abspath("datasets/fer2013")
train_data_path = os.path.join(dataset_path, "train")
test_data_path = os.path.join(dataset_path, "test")

# Define the parameters for training and testing
batch_size = 32
num_epochs = 10
image_height, image_width = 48, 48
num_classes = 7  # 7 emotions: angry, disgust, fear, happy, neutral, sad, surprise

# Create data generators for training and testing
train_data_generator = ImageDataGenerator(rescale=1./255)
test_data_generator = ImageDataGenerator(rescale=1./255)

train_generator = train_data_generator.flow_from_directory(
    train_data_path,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)

test_generator = test_data_generator.flow_from_directory(
    test_data_path,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)
# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(image_height, image_width, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=num_epochs,
    validation_data=test_generator,
    validation_steps=test_generator.n // batch_size
)

# Evaluate the model
score = model.evaluate_generator(test_generator, steps=test_generator.n // batch_size)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
