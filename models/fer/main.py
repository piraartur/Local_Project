import cv2

from fer import FER

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

if __name__ == "__main__":
    input_image = cv2.imread("C:/Users/karol/Desktop/Studia/6 Semestr/ProjektLokalny/Local_Project/datasets/fer2013/test/angry/PrivateTest_3411628.jpg")
    emotion_detector = FER()
    # Output image's information
    print(emotion_detector.detect_emotions(input_image))

