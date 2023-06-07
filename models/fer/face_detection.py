import os
import cv2
from natsort import natsorted

from fer import FER

from models.fer.data_manipulation import (
    get_dominant_emotion_from_result,
    check_model_detection_rate,
    calculate_model_percentage_detection_rate,
    get_emotions_from_result,
    combine_emotion_values_from_all_images,
    calculate_model_total_emotions_percentages,
    write_emotions_to_csv_file,
)
from models.fer.helpers import (
    get_absolute_folder_path,
    get_emotion_from_folder_path,
)


def fer():
    detection_rate_angry, not_detected_angry = detect_emotions_and_write_to_csv_file(emotion="angry")
    detection_rate_disgust, not_detected_disgust = detect_emotions_and_write_to_csv_file(emotion="disgust")
    detection_rate_fear, not_detected_fear = detect_emotions_and_write_to_csv_file(emotion="fear")
    detection_rate_happy, not_detected_happy = detect_emotions_and_write_to_csv_file(emotion="happy")
    detection_rate_neutral, not_detected_neutral = detect_emotions_and_write_to_csv_file(emotion="neutral")
    detection_rate_sad, not_detected_sad = detect_emotions_and_write_to_csv_file(emotion="sad")
    detection_rate_surprise, not_detected_surprise = detect_emotions_and_write_to_csv_file(emotion="surprise")

    emotions = {
        "angry": detection_rate_angry,
        "disgust": detection_rate_disgust,
        "fear": detection_rate_fear,
        "happy": detection_rate_happy,
        "neutral": detection_rate_neutral,
        "sad": detection_rate_sad,
        "surprise": detection_rate_surprise,
    }
    emotions_not_detected = {
        "angry": not_detected_angry,
        "disgust": not_detected_disgust,
        "fear": not_detected_fear,
        "happy": not_detected_happy,
        "neutral": not_detected_neutral,
        "sad": not_detected_sad,
        "surprise": not_detected_surprise,
    }

    write_emotions_to_csv_file(
        emotions=emotions, csv_file_name=f"emotions_fer.csv"
    )
    write_emotions_to_csv_file(
        emotions=emotions_not_detected, csv_file_name=f"emotions_not_detected_fer.csv"
    )


def detect_emotions_and_write_to_csv_file(emotion):
    detection_rate, emotions, not_detected = detect_folder_images_face_expressions(
        f"datasets/fer2013/test/{emotion}/"
    )
    write_emotions_to_csv_file(
        emotions=emotions, csv_file_name=f"{emotion}_fer.csv"
    )
    return detection_rate, not_detected


def fer_face_expression_detection(file_path):
    input_image = cv2.imread(file_path)

    emotion_detector = FER(mtcnn=True)

    result = emotion_detector.detect_emotions(input_image)
    # display_image(input_image)

    return result


def detect_folder_images_face_expressions(root_folder_path):
    absolute_folder_path = get_absolute_folder_path(root_folder_path=root_folder_path)
    for root, dirs, file_names in os.walk(absolute_folder_path):
        file_names = natsorted(file_names)
        detection_rate, emotions, not_detected = detect_single_image_face_expression(
            file_names=file_names, absolute_folder_path=absolute_folder_path
        )
        return detection_rate, emotions, not_detected


def detect_single_image_face_expression(file_names, absolute_folder_path):
    correct_detections = 0
    emotions_not_detected = 0
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
        result = fer_face_expression_detection(absolute_file_path)
        emotion = get_emotion_from_folder_path(
            absolute_folder_path=absolute_folder_path
        )

        dominant_emotion = get_dominant_emotion_from_result(result=result)
        emotions, emotions_not_detected = get_emotions_from_result(result=result,
                                                                   emotions_not_detected=emotions_not_detected)

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
        file_names=file_names, emotions=emotions, emotions_not_detected=emotions_not_detected
    )
    detection_rate, not_detected = calculate_model_percentage_detection_rate(
        correct_detections=correct_detections, file_names=file_names, emotions_not_detected=emotions_not_detected
    )
    return detection_rate, emotions_final, not_detected
