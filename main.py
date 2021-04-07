import cv2
import numpy as np
import cv2.aruco as aruco
import object_size
import transform
import aruco_finder
from scipy.spatial import distance as dist

if __name__ == '__main__':
    img = cv2.imread('./input/Mtest2.jpg', cv2.COLOR_BGR2GRAY)
    marker = aruco_finder.findAruco(img)

    trans_image = transform.four_point_transform(img, marker)
    cv2.imshow("transformed", trans_image)
    cv2.imwrite("./output/transformed.jpg", trans_image)

    trans_marker = aruco_finder.findAruco(trans_image)
    pixelsPerMetric = dist.euclidean(trans_marker[2], trans_marker[1]) / 8
    print(pixelsPerMetric)

    # pass the rotated image as well as the bottom right coordinates of the aruco marker
    object_size.objectsize(trans_image, trans_marker[0], pixelsPerMetric)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


