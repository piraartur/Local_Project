import os
from pathlib import Path

import matplotlib.pyplot as plt


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def get_absolute_folder_path(root_folder_path):
    root = get_project_root()
    absolute_folder_path = os.path.join(root, root_folder_path)
    return absolute_folder_path


def get_emotion_from_folder_path(absolute_folder_path):
    emotion_index = get_index_of_emotion_from_folder_path(absolute_folder_path=absolute_folder_path)
    emotion = ""
    for index, char in enumerate(reversed(absolute_folder_path)):
        if index < emotion_index and char != "/":
            emotion += char
    return emotion[::-1]


def get_index_of_emotion_from_folder_path(absolute_folder_path):
    for index, char in enumerate(reversed(absolute_folder_path)):
        if char == "/" and index != 0:
            return index


def display_image(image):
    plt.imshow(image[:, :, ::-1])
    plt.show()
