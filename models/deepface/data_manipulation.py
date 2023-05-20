def get_dominant_emotion_from_result(result):
    dominant_emotion = result[0]['dominant_emotion']
    return dominant_emotion


def check_model_detection_rate(dominant_emotion, emotion, correct_detections):
    if dominant_emotion == emotion:
        correct_detections += 1
    return correct_detections


def calculate_model_percentage_detection_rate(correct_detections, file_names):
    return correct_detections/len(file_names)
