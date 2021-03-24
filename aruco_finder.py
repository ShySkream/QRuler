import cv2
import numpy as np
import cv2.aruco as aruco
import object_size

if __name__ == '__main__':
    aruco_list = {}

    # Read image
    img_w_aruco = cv2.imread('./input/Mtest1.jpg', cv2.COLOR_BGR2GRAY)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

    # Marker detection parameters
    parameters = aruco.DetectorParameters_create()

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img_w_aruco, aruco_dict, parameters=parameters)

    # Can subtract # of pixels to determine how long 5 cm is and use it to ratio
    print(corners)

    image = img_w_aruco.copy()
    aruco.drawDetectedMarkers(image, corners)

    """
    I do not have the distortion coefficient... not sure how to get that. Seems like something to do with the camera
    distortion_coefficient = np.array([2.0426196677407879e-01, -3.3902097431574091e-01, -4.1813964792274307e-03, -1.0425257413809015e-02, 8.2004709580884308e-02])

    We would need the distortion_coefficient to get the rvec and tvec. With rvec and tvec, we can draw the axis (x,y,z)

    rvec = aruco.estimatePoseSingleMarkers(corners, 0.05, image, np.array([]))
    print(rvec)

    Think we should switch to using a webcam for this.. would make our lives eaasier. We can point the webcam at the object (desk or something)
    """

    cv2.imshow("name", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # pass the rotated image as well as the bottom right coordinates of the aruco marker
    # TODO pass rotated image instead of original
    object_size.objectsize(img_w_aruco, corners[0][0][0])


