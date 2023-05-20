import os
import cv2
from natsort import natsorted

from deepface import DeepFace

from models.deepface.data_manipulation import get_dominant_emotion_from_result, check_model_detection_rate, \
    calculate_model_percentage_detection_rate
from models.deepface.helpers import (
    get_project_root,
    display_image,
    get_absolute_folder_path, get_emotion_from_folder_path,
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
    correct_detections = 0
    for file_name in file_names:
        absolute_file_path = absolute_folder_path + file_name
        result = deepface_face_expression_detection(absolute_file_path)
        dominant_emotion = get_dominant_emotion_from_result(result=result)
        emotion = get_emotion_from_folder_path(absolute_folder_path=absolute_folder_path)
        correct_detections = check_model_detection_rate(dominant_emotion=dominant_emotion, emotion=emotion,
                                                        correct_detections=correct_detections)
    percentage_detection_rate = calculate_model_percentage_detection_rate(correct_detections=correct_detections,file_names=file_names)
    print(percentage_detection_rate)
