def get_dominant_emotion_from_result(result):
    dominant_emotion = result[0]['dominant_emotion']
    return dominant_emotion


def check_model_detection_rate(dominant_emotion, emotion, correct_detections):
    if dominant_emotion == emotion:
        correct_detections += 1
    return correct_detections


