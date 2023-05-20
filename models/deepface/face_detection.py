import os
import cv2
from natsort import natsorted

from deepface import DeepFace

from models.deepface.data_manipulation import (
    get_dominant_emotion_from_result,
    check_model_detection_rate,
    calculate_model_percentage_detection_rate,
    get_emotions_from_result,
    combine_emotion_values_from_all_images,
    calculate_model_total_emotions_percentages,
)
from models.deepface.helpers import (
    get_project_root,
    display_image,
    get_absolute_folder_path,
    get_emotion_from_folder_path,
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
        detection_rate, emotions = detect_single_image_face_expression(
            file_names=file_names, absolute_folder_path=absolute_folder_path
        )
        return detection_rate, emotions


def detect_single_image_face_expression(file_names, absolute_folder_path):
    correct_detections = 0
    emotions_old = {
        "angry": 0,
        "disgust": 0,
        "fear": 0,
        "happy": 0,
        "sad": 0,
        "surprise": 0,
        "neutral": 0,
    }
    for file_name in file_names:
        absolute_file_path = absolute_folder_path + file_name
        result = deepface_face_expression_detection(absolute_file_path)
        emotion = get_emotion_from_folder_path(
            absolute_folder_path=absolute_folder_path
        )

        dominant_emotion = get_dominant_emotion_from_result(result=result)
        emotions = get_emotions_from_result(result=result)

        emotions = combine_emotion_values_from_all_images(
            dict1=emotions, dict2=emotions_old
        )
        correct_detections = check_model_detection_rate(
            dominant_emotion=dominant_emotion,
            emotion=emotion,
            correct_detections=correct_detections,
        )
        emotions_old = emotions

    emotions_final = calculate_model_total_emotions_percentages(
        file_names=file_names, emotions=emotions
    )
    detection_rate = calculate_model_percentage_detection_rate(
        correct_detections=correct_detections, file_names=file_names
    )

    return detection_rate, emotions_final
