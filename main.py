import sys

sys.path.append('./')
sys.path.append('./src')
from yolo.net.yolo_tiny_net import YoloTinyNet
import tensorflow as tf
import cv2
import numpy as np
import nonmax
import video
from common import anorm2, draw_str
from time import clock

classes_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


lk_params = dict( winSize  = (20, 20),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

video_src = 'videos/traffic4.mp4'


class TrackingObject:
    def __init__(self, id, position, class_num):
        self.threshold = 3
        self.maxBalance = 6
        self.id = id
        self.size = [50, 50]
        self.position = position
        self.class_num = class_num
        self.balance = 1
        self.realObj = False

    def addbalance(self):
        self.balance = min(self.balance + 1, self.maxBalance)


    def reducebalance(self):
        self.balance = max(self.balance - 0.5, 0)


def isInsideBox(box, position, percent):
    xMin = box[0] + (box[2] - box[0]) *(1 - percent )/2
    xMax = box[2] - (box[2] - box[0]) *(1 - percent )/2
    yMin = box[1] + (box[3] - box[1]) *(1 - percent )/2
    yMax = box[3] - (box[3] - box[1]) *(1 - percent )/2

    return position[0] > xMin and position[0] < xMax and position[1] > yMin and position[1] < yMax



def process_predicts(predicts):
    p_classes = predicts[0, :, :, 0:20]
    C = predicts[0, :, :, 20:22]
    coordinate = predicts[0, :, :, 22:]

    p_classes = np.reshape(p_classes, (7, 7, 1, 20))
    C = np.reshape(C, (7, 7, 2, 1))

    P = C * p_classes

    #  print P[5,1, 0, :]

    thr = 0.07

# index = np.argmax(P)
    index = np.nonzero(P >= thr)
    prob = P[index]
    class_num = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for i in range(len(index[0])):
        # index = np.unravel_index(index, P.shape)

        class_num.append(index[3][i])

        coordinate = np.reshape(coordinate, (7, 7, 2, 4))
        max_coordinate = coordinate[index[0][i], index[1][i], index[2][i], :]

        xcenter = max_coordinate[0]
        ycenter = max_coordinate[1]
        w = max_coordinate[2]
        h = max_coordinate[3]

        xcenter = (index[1][i] + xcenter) * (448 / 7.0)
        ycenter = (index[0][i] + ycenter) * (448 / 7.0)

        w = w * 448
        h = h * 448

        xmin.append(xcenter - w / 3.0)
        ymin.append(ycenter - h / 3.0)

        xmax.append(xcenter + w / 3.0)
        ymax.append(ycenter + h / 3.0)

    return xmin, ymin, xmax, ymax, class_num, prob


# Set up net parameters and solver
common_params = {'image_size': 448, 'num_classes': 20,
                 'batch_size': 1}
net_params = {'cell_size': 7, 'boxes_per_cell': 2, 'weight_decay': 0.0005}

net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, 448, 448, 3))
predicts = net.inference(image)

sess = tf.Session()

# Read saved model
saver = tf.train.Saver(net.trainable_collection)
saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')


# Read video file
cap = cv2.VideoCapture(video_src)
print cap.isOpened()
#np_img = cv2.imread('cat.jpg')
frame_idx = 0


# tracking parameters

track_len = 10
detect_interval = 5


# main loop
prev_box = []

trackingObjects = []
totalObjects = np.zeros(20)


while(cap.isOpened()):
    print "frame:", frame_idx
    ret, original = cap.read()
    print original.shape
    crop_img = original[300:720, 400:820]
    # crop_img = original
    resized_img = cv2.resize(crop_img, (448, 448))
    np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    vis = resized_img.copy()

    np_img = np_img.astype(np.float32)

    np_img = np_img / 255.0 * 2 - 1
    np_img = np.reshape(np_img, (1, 448, 448, 3))

    np_predict = sess.run(predicts, feed_dict={image: np_img})

    xmin, ymin, xmax, ymax, class_num, prob = process_predicts(np_predict)

    sortedIndex = sorted(range(len(prob)), key=lambda k: prob[k], reverse=True)
    prob = sorted(prob, reverse=True)
    boxes = zip(xmin, ymin, xmax, ymax)
    boxes = np.array([boxes[i] for i in sortedIndex])
    print "box_before", len(boxes)
    boxes, index_suppress = nonmax.non_max_suppression(boxes, 0.4)
    print "box_after", len(boxes)
    classes = [class_num[i] for i in index_suppress]

    # print "number of boxes:", len(xmin)
    # print "Class:", classes
    # print "Prob: ", prob
    # print "boxes:", boxes
    # print "index:", index_suppress

    for n in range(len(boxes)):
        cv2.rectangle(vis, (int(boxes[n, 0]),
            int(boxes[n, 1])), (int(boxes[n, 2]), int(boxes[n, 3])), (0, 0, 255))

    if frame_idx > 0 and len(trackingObjects) != 0:
        # tracking:

        tracks = [];

        for n in range(len(trackingObjects)):
            pos = trackingObjects[n].position
            tracks.append(pos)

        img0, img1 = prev_gray, frame_gray

        p0 = np.float32([tr[:] for tr in tracks]).reshape(-1, 1, 2)

        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 10
        new_tracks = []

        newtrackingObj = []

        for n in range(len(trackingObjects)):
            if good[n]:
                trackingObjects[n].position = p1[n].tolist()[0]
                newtrackingObj.append(trackingObjects[n])
        trackingObjects = newtrackingObj

        # # draw tracking results
        # for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
        #     tr.append((x, y))
        #     if x > 400:
        #         continue
        #
        #     new_tracks.append([x, y])
        #     # cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)
        #     cv2.rectangle(vis, (int(x - 10), int(y - 10)), (int(x + 10), int(y + 10)), (0, 255, 255))
        #     # cv2.putText(vis, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))

        ## end tracking


    prev_gray = frame_gray

    # update trackingObjects
    if len(trackingObjects) == 0:
        for n in range(len(boxes)):
            objId = np.sum(totalObjects)
            position = [(boxes[n,0]+boxes[n,2])/2, (boxes[n,1]+boxes[n,3])/2 ]
            class_num = classes[n]
            newobject = TrackingObject(objId, position, class_num)
            newobject.size = [(boxes[n,2]-boxes[n,0]), (boxes[n,3]+boxes[n,1]) ]
            # totalObjects[class_num] = totalObjects[class_num] + 1

            trackingObjects.append(newobject)
    else:
        boxFind = np.zeros(len(boxes))
        newtrackingObj = []
        for n in range(len(trackingObjects)):
            currentObj = trackingObjects[n]
            position = currentObj.position
            foundBoxes = False

            for m in range(len(boxes)):
                box = boxes[m]
                #if isInsideBox(box, position, 1) and classes[m] == currentObj.class_num:
                if isInsideBox(box, position, 1):

                    # currentObj.position = [(box[0]+box[2])/2, (box[1]+box[3])/2]
                    currentObj.addbalance()
                    currentObj.size = [box[2] - box[0], (box[3] - box[1])]
                    if currentObj.realObj == False and currentObj.balance >= currentObj.threshold:
                        currentObj.realObj = True
                        totalObjects[currentObj.class_num] = totalObjects[currentObj.class_num]  + 1
                    boxFind[m] = 1
                    foundBoxes = True
                    break
            if ~foundBoxes:
                currentObj.reducebalance()
            if currentObj.balance > 0:
                newtrackingObj.append(currentObj)

        trackingObjects = newtrackingObj

        for n in range(len(boxes)):
            if boxFind[n] == 0:
                objId = np.sum(totalObjects)
                position = [(boxes[n,0]+boxes[n,2])/2, (boxes[n,1]+boxes[n,3])/2 ]
                class_num = classes[n]
                newobject = TrackingObject(objId, position, class_num)
                newobject.size = [(boxes[n,2]-boxes[n,0]), (boxes[n,3]-boxes[n,1])]
                # totalObjects[class_num] = totalObjects[class_num] + 1

                trackingObjects.append(newobject)


    # draw:
    for i in range(len(trackingObjects)):
        thisObj = trackingObjects[i]
        print thisObj.realObj
        if thisObj.realObj == True:
            class_name = classes_name[thisObj.class_num]
            pos = thisObj.position
            size = thisObj.size
            cv2.rectangle(vis, (int(pos[0] - size[0]/2),
                int(pos[1] -  size[1]/2)), (int(pos[0] + size[0]/2), int(pos[1] + size[1]/2)), (0, 255, 255))
        #    cv2.putText(resized_img, class_name[i], (int(xmin[i]), int(ymin[i])), 2, 1.5, (0, 0, 255))

    cv2.imshow('cars', vis)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    print "Sum of tracked objects:", np.sum(totalObjects)

    frame_idx = frame_idx + 1

cap.release()
cv2.destroyAllWindows()

sess.close()
