import csv


def get_dominant_emotion_from_result(result):
    try:
        emotions = result[0]["emotions"]
        max_value = max(emotions.values())
        keys_with_max_value = [
            key for key, value in emotions.items() if value == max_value
        ]
        if len(keys_with_max_value) > 1:
            return keys_with_max_value
        else:
            return keys_with_max_value[0]
    except IndexError:
        pass


def get_emotions_from_result(result, emotions_not_detected):
    if not result:
        emotions_not_detected += 1
        emotions = {}
    else:
        emotions = result[0]["emotions"]
    return emotions, emotions_not_detected


def combine_emotion_values_from_all_images(dict1, dict2):
    result = {
        key: dict1.get(key, 0) + dict2.get(key, 0) for key in set(dict1) | set(dict2)
    }
    return result


def check_model_detection_rate(dominant_emotion, emotion, correct_detections):
    if dominant_emotion == emotion and dominant_emotion is not None:
        correct_detections += 1
    return correct_detections


def calculate_model_percentage_detection_rate(
    correct_detections, file_names, emotions_not_detected
):
    detection_rate = correct_detections / (len(file_names) - emotions_not_detected)
    not_detected = emotions_not_detected / len(file_names)
    return detection_rate, not_detected


def calculate_model_total_emotions_percentages(
    emotions, file_names, emotions_not_detected
):
    return {
        key: value / (len(file_names) - emotions_not_detected)
        for key, value in emotions.items()
    }


def write_emotions_to_csv_file(emotions, csv_file_name):
    with open(csv_file_name, "w") as f:
        w = csv.DictWriter(f, emotions.keys())
        w.writeheader()
        w.writerow(emotions)
