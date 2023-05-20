import time

from models.deepface.face_detection import (
    deepface_face_expression_detection,
    detect_folder_images_face_expressions,
)

if __name__ == "__main__":
    start_time = time.time()

    #result = deepface_face_expression_detection("C:/Users/karol/Desktop/Studia/Projects/Local_Project/datasets/fer2013/test/angry/PrivateTest_88305.jpg")
    #print(result)
    #result = [{'emotion': {'angry': 46.04460000991821, 'disgust': 9.36625212943909e-05, 'fear': 8.291973918676376, 'happy': 0.0001523979790363228, 'sad': 38.99666368961334, 'surprise': 4.050834547797422e-05, 'neutral': 6.666478514671326}, 'dominant_emotion': 'angry', 'region': {'x': 0, 'y': 0, 'w': 48, 'h': 48}}]
    #print((result[0]['dominant_emotion']))

    detect_folder_images_face_expressions("datasets/fer2013/test/sad/")

    print("--- %s seconds ---" % (time.time() - start_time))
