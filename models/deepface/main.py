import time


from models.deepface.face_detection import deepface

if __name__ == "__main__":
    start_time = time.time()

    deepface()

    print("--- %s seconds ---" % (time.time() - start_time))
