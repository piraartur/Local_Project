import time

from models.deepface.face_detection import (
    deepface_face_expression_detection,
    detect_folder_images_face_expressions,
)

if __name__ == "__main__":
    start_time = time.time()

    # result = detect_face_expression()
    # print(result)
    detect_folder_images_face_expressions("datasets/fer2013/test/angry_test/")

    print("--- %s seconds ---" % (time.time() - start_time))
