import time

from models.deepface.face_detection import detect_folder_images_face_expressions

if __name__ == "__main__":
    start_time = time.time()

    detection_rate, emotions = detect_folder_images_face_expressions(
        "datasets/fer2013/test/sad/"
    )
    print(detection_rate)
    print(emotions)

    print("--- %s seconds ---" % (time.time() - start_time))
