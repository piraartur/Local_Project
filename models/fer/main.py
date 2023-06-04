import time

from models.fer.face_detection import fer

if __name__ == "__main__":
    start_time = time.time()

    fer()

    print("--- %s seconds ---" % (time.time() - start_time))
