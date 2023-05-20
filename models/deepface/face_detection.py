import os
import cv2
from natsort import natsorted

from deepface import DeepFace

from models.deepface.helpers import (
    get_project_root,
    display_image,
    get_absolute_folder_path,
)


def deepface_face_expression_detection(file_path):
    input_image = cv2.imread(file_path)

    result = DeepFace.analyze(input_image, actions=["emotion"], enforce_detection=False)
    # display_image(input_image)

    return result


def detect_folder_images_face_expressions(root_folder_path):
    absolute_folder_path = get_absolute_folder_path(root_folder_path=root_folder_path)
    for root, dirs, file_names in os.walk(absolute_folder_path):
        file_names = natsorted(file_names)
        detect_single_image_face_expression(
            file_names=file_names, absolute_folder_path=absolute_folder_path
        )


def detect_single_image_face_expression(file_names, absolute_folder_path):
    for file_name in file_names:
        absolute_file_path = absolute_folder_path + file_name
        result = deepface_face_expression_detection(absolute_file_path)
        print(result)
