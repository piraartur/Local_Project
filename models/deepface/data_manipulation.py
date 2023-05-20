def get_dominant_emotion_from_result(result):
    dominant_emotion = result[0]["dominant_emotion"]
    return dominant_emotion


def get_emotions_from_result(result):
    emotions = result[0]["emotion"]
    return emotions


def combine_emotion_values_from_all_images(dict1, dict2):
    result = {
        key: dict1.get(key, 0) + dict2.get(key, 0) for key in set(dict1) | set(dict2)
    }
    return result


def check_model_detection_rate(dominant_emotion, emotion, correct_detections):
    if dominant_emotion == emotion:
        correct_detections += 1
    return correct_detections


def calculate_model_percentage_detection_rate(correct_detections, file_names):
    return correct_detections / len(file_names)


def calculate_model_total_emotions_percentages(emotions, file_names):
    return {key: value / len(file_names) for key, value in emotions.items()}
