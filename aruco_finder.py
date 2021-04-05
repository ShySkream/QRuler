import cv2
import numpy as np
import cv2.aruco as aruco
import object_size
import transform

def findAruco(img):
    aruco_list = {}

    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

    # Marker detection parameters
    parameters = aruco.DetectorParameters_create()

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    # Can subtract # of pixels to determine how long 5 cm is and use it to ratio
    print(corners)

    image = img.copy()
    aruco.drawDetectedMarkers(image, corners)
    # cv2.circle(image, (corners[0][0][3][0],corners[0][0][3][1]), 15, (0, 255, 0), -1)
    cv2.imshow("name", image)

    return corners[0][0]

    cv2.waitKey(0)
    cv2.destroyAllWindows()


