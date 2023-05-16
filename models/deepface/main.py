#import the required modules
import cv2
import matplotlib.pyplot as plt
import time
from deepface import DeepFace

if __name__ == "__main__":
    start_time = time.time()
    input_image = cv2.imread("C:/Users/karol/Desktop/Studia/6 Semestr/ProjektLokalny/Local_Project/datasets/fer2013/test/angry/PrivateTest_3411628.jpg")
    plt.imshow(input_image[:, :, :: -1])

    # display that image
    plt.show()

    result = DeepFace.analyze(input_image, actions=['emotion'],enforce_detection=False)


    # print result
    print(result)
    print("--- %s seconds ---" % (time.time() - start_time))


