import cv2
import numpy as np
import cv2.aruco as aruco


def find_aruco_marker(img):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

    # Marker detection parameters
    parameters = aruco.DetectorParameters_create()

    # Lists of ids and the corners belonging to each id
    corners, ids, rejected_points = aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    # Draw what the aruco detector sees
    image = img.copy()
    aruco.drawDetectedMarkers(image, corners)
    cv2.imwrite("./output/Marker.jpg", image)

    # return only the first marker on the list
    return corners[0][0]


def get_calibration_data():
    """
    The distortion coefficient retrieved is for a specific camera; images taken on another camera may produce different results.
    """
    # Read Yaml file and retrieve distortion coefficient and camera matrix
    calibration_file = cv2.FileStorage("./calibration.yaml", cv2.FileStorage_READ)
    file_node = calibration_file.getNode('camera_matrix')
    matrix = np.asarray(file_node.mat())
    file_node = calibration_file.getNode('dist_coeff')
    distortion = np.asarray(file_node.mat())

    return matrix, distortion


def calibrate_image(img, matrix, distortion, new_width=1280, new_height=960):

    # start by scaling all input images
    new_img = cv2.resize(img, (new_width, new_height))

    # Get height/width of image
    h, w = new_img.shape[:2]

    # commented out due to bad warping of objects.
    # new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w, h), 1, (w, h))
    # undistorted_img = cv2.undistort(new_img, matrix, distortion, None, new_camera_mtx)

    undistorted_img = new_img

    # Crop the image to remove the black border (warped distortion)
    # this cropping is too aggressive for our needs, so it is left commented out.
    # x, y, width, height = roi
    # cropped_img = undistorted_img[y:y+height, x:x+width]

    return undistorted_img

