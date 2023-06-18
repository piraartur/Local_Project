import time

from face_detection import fer, detect_rotated_images

if __name__ == "__main__":
    start_time = time.time()

    # fer()
    file_path = "/Users/sold/Desktop/Python/Projects/University/Local_Project/datasets/fer2013/test/angry/PrivateTest_88305.jpg"
    detect_rotated_images(file_path=file_path)

    print("--- %s seconds ---" % (time.time() - start_time))
