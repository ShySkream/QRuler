import cv2
import object_size
import transform
import argparse
import aruco_finder
from scipy.spatial import distance as dist

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    args = vars(ap.parse_args())

    # load the image
    img = cv2.imread(args["image"], cv2.COLOR_BGR2GRAY)

    # get and process the camera calibration to warp the image
    matrix, distortion = aruco_finder.get_calibration_data()
    calibrated_img = aruco_finder.calibrate_image(img, matrix, distortion)

    # find the marker in the image
    marker = aruco_finder.find_aruco_marker(calibrated_img)

    # transform the image to be bird's eye view
    trans_image = transform.four_point_transform(calibrated_img, marker)
    cv2.imwrite("./output/transformed.jpg", trans_image)

    # find the location of the marker in the transformed image
    trans_marker = aruco_finder.find_aruco_marker(trans_image)

    # calculate the pixels per metric for the image.  This represents how many pixels are in a cm.
    pixelsPerMetric = dist.euclidean(trans_marker[2], trans_marker[1]) / 8
    print("PPM: ", pixelsPerMetric)

    # pass the rotated image as well as the bottom right coordinates of the aruco marker
    object_size.objectsize(trans_image, trans_marker[0], pixelsPerMetric)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


