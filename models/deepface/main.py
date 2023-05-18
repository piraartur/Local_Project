# import the required modules
import os

import cv2

import time
from deepface import DeepFace

from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


if __name__ == "__main__":
    start_time = time.time()
    root = get_project_root()
    input_image = cv2.imread(os.path.join(root, "datasets/fer2013/test/angry/PrivateTest_88305.jpg"))
    # plt.imshow(input_image[:, :, :: -1])

    # display that image
    # plt.show()
    result = DeepFace.analyze(input_image, actions=['emotion'], enforce_detection=False)

    # print result
    print(result)
    print("--- %s seconds ---" % (time.time() - start_time))
