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
    write_emotions_to_csv_file,
)
from models.deepface.helpers import (
    get_absolute_folder_path,
    get_emotion_from_folder_path,
)


def deepface():
    detection_rate_angry = detect_emotions_and_write_to_csv_file(emotion="angry")
    detection_rate_disgust = detect_emotions_and_write_to_csv_file(emotion="disgust")
    detection_rate_fear = detect_emotions_and_write_to_csv_file(emotion="fear")
    detection_rate_happy = detect_emotions_and_write_to_csv_file(emotion="happy")
    detection_rate_neutral = detect_emotions_and_write_to_csv_file(emotion="neutral")
    detection_rate_sad = detect_emotions_and_write_to_csv_file(emotion="sad")
    detection_rate_surprise = detect_emotions_and_write_to_csv_file(emotion="surprise")

    emotions = {
        "angry": detection_rate_angry,
        "disgust": detection_rate_disgust,
        "fear": detection_rate_fear,
        "happy": detection_rate_happy,
        "neutral": detection_rate_neutral,
        "sad": detection_rate_sad,
        "surprise": detection_rate_surprise,
    }
    write_emotions_to_csv_file(
        emotions=emotions, csv_file_name=f"emotions_deepface.csv"
    )


def detect_emotions_and_write_to_csv_file(emotion):
    detection_rate, emotions = detect_folder_images_face_expressions(
        f"datasets/fer2013/test/{emotion}/"
    )
    write_emotions_to_csv_file(
        emotions=emotions, csv_file_name=f"{emotion}_deepface.csv"
    )
    return detection_rate


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


def detect_rotated_images(file_path):
    print(f"Not rotated: {deepface_face_expression_detection(file_path=file_path)}")
    detect_rotated_90(file_path=file_path)
    detect_rotated_180(file_path=file_path)
    detect_rotated_270(file_path=file_path)


def detect_rotated_90(file_path):
    src = cv2.imread(file_path)
    image = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
    result = DeepFace.analyze(image, actions=["emotion"], enforce_detection=False)
    print(f"Rotated 90: {result}")


def detect_rotated_180(file_path):
    src = cv2.imread(file_path)
    image = cv2.rotate(src, cv2.ROTATE_180)
    result = DeepFace.analyze(image, actions=["emotion"], enforce_detection=False)
    print(f"Rotated 180: {result}")


def detect_rotated_270(file_path):
    src = cv2.imread(file_path)
    image = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)
    result = DeepFace.analyze(image, actions=["emotion"], enforce_detection=False)
    print(f"Rotated 270: {result}")
