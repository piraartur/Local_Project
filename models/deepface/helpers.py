import os
from pathlib import Path

import matplotlib.pyplot as plt


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def get_absolute_folder_path(root_folder_path):
    root = get_project_root()
    absolute_folder_path = os.path.join(root, root_folder_path)
    return absolute_folder_path


def display_image(image):
    plt.imshow(image[:, :, ::-1])
    plt.show()
