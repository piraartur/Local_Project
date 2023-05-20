import os
import cv2

from deepface import DeepFace

from models.deepface.helpers import get_project_root, display_image


def detect_face_expression():
    root = get_project_root()
    input_image = cv2.imread(os.path.join(root, "datasets/fer2013/test/angry/PrivateTest_88305.jpg"))

    result = DeepFace.analyze(input_image, actions=['emotion'], enforce_detection=False)
    display_image(input_image)

    return result
