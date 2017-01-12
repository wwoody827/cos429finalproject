# import the necessary packages
import numpy as np


# our own nonmax

def non_max_suppression(boxes, overlapThresh):
	if len(boxes) == 0:
		return [], []

	index_remain = np.ones(len(boxes))

	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)

	for n in range(len(boxes) - 1):
		if index_remain[n] == 0:
			continue

		for m in range(n + 1, len(boxes)):
			if index_remain[m] == 0:
				continue


			XA1 = boxes[n,0]
			YA1 = boxes[n,1]
			XA2 = boxes[n,2]
			YA2 = boxes[n,3]

			XB1 = boxes[m,0]
			YB1 = boxes[m,1]
			XB2 = boxes[m,2]
			YB2 = boxes[m,3]

			SI= max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))


			SU = min(area[n], area[m])
			print "SI:", SI, "SU:", SU
			overlap = 1.0* SI/SU

			if overlap > overlapThresh:
				index_remain[m] = 0


	index_pick = np.nonzero(index_remain)
	index_pick = index_pick[0]

	box_suppressed = boxes[np.array(index_pick), :]

	return box_suppressed, index_pick



# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# indexList = np.arrange(len(boxes))
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")
