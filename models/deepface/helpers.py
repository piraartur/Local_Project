from pathlib import Path

import matplotlib.pyplot as plt


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def display_image(image):
    plt.imshow(image[:, :, :: -1])
    plt.show()
