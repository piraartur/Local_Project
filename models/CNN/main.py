import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

# Get the absolute paths of the dataset and model files
dataset_path = os.path.abspath("datasets/fer2013")
train_data_path = os.path.join(dataset_path, "train")
test_data_path = os.path.join(dataset_path, "test")

# Define the parameters for training and testing
batch_size = 32
num_epochs = 1
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
    batch_size=1,  # Set batch_size to 1 for predictions on individual images
    shuffle=False,  # Disable shuffling to ensure predictions match the original order
    color_mode="grayscale",
    class_mode=None  # Set class_mode to None to avoid loading labels
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
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=num_epochs
)

# Evaluate the model
score = model.evaluate(test_generator, steps=test_generator.samples)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Get the predictions on the test set
predictions = model.predict(test_generator, steps=test_generator.samples, verbose=1)

# Get the filenames of the original test images
original_filenames = [filename.split('/')[-1] for filename in test_generator.filenames]

# Create a DataFrame with the predictions and filenames
results_df = pd.DataFrame(predictions, columns=[str(i) for i in range(num_classes)])
results_df['Filename'] = original_filenames

# Group the predictions by emotion and calculate the average accuracy
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_accuracies = []
for label in emotion_labels:
    label_indices = [i for i, filename in enumerate(original_filenames) if label in filename]
    label_predictions = predictions[label_indices]
    label_average = np.mean(label_predictions, axis=0)
    label_accuracy = label_average[emotion_labels.index(label)] / np.sum(label_average)
    emotion_accuracies.append(label_accuracy)
    label_results_df = pd.DataFrame(label_predictions, columns=[str(i) for i in range(num_classes)])
    label_results_df['Filename'] = [original_filenames[i] for i in label_indices]
    label_results_df.to_csv(f'./models/CNN/csv_files/{label}.csv', index=False)

# Create a DataFrame with the emotion accuracies and emotion labels
emotion_results_df = pd.DataFrame({'Emotion': emotion_labels, 'Accuracy': emotion_accuracies})

# Transpose the DataFrame to have emotions as columns and save it to a CSV file
emotion_results_df.set_index('Emotion').transpose().to_csv('./models/CNN/csv_files/emotion_results.csv')
