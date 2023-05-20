import time

from models.deepface.face_detection import detect_face_expression


if __name__ == "__main__":
    start_time = time.time()

    result = detect_face_expression()
    print(result)

    print("--- %s seconds ---" % (time.time() - start_time))
