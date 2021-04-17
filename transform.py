# import the necessary packages
import numpy as np
import cv2


def removeEmpty(img):
	# this function removes pitch black space created when the transformed image
	# generates empty space not part of the original.
	h, w, _ = img.shape
	print(h, w)
	cv2.imwrite("./output/no crop.jpg", img)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imwrite("./output/no crop bw.jpg", gray)
	minx = 9999999
	maxx = 0
	miny = 9999999
	maxy = 0

	for y in range(h-1):
		for x in range(w-1):
			if gray[y, x] != 0:
				if miny == 9999999:
					miny = y
				if maxx < x:
					maxx = x
				if minx > x:
					minx = x
				maxy = y

	print(miny, maxy, minx, maxx)

	img2 = img[miny+10:maxy-10, minx+10:maxx-10]
	return img2


def GetPadding(img, pts):
	height, width, _ = img.shape
	return height, height, width, width
	# 310, 3400, 640, 700


def four_point_transform(image, pts):
	# Calculate the space needed on all sides of the aruco marker
	padUp, padDown, padLeft, padRight = GetPadding(image, pts)
	rect = pts
	(tl, tr, br, bl) = rect
	print(rect)

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	maxSize = max(int(maxHeight), int(maxWidth))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[padLeft, padUp],
		[padLeft+maxSize, padUp],
		[padLeft+maxSize, padUp+maxSize],
		[padLeft, padUp+maxSize]], dtype="float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	print(M)
	# Apply perspective transform to the image
	warped = cv2.warpPerspective(image, M, (maxSize+padRight+padLeft, maxSize+padDown+padUp))
	# remove empty space added by the warp
	warped = removeEmpty(warped)
	# return the warped image
	return warped
