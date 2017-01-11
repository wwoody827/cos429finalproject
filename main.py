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


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


video_src = 'videos/traffic.mp4'

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

        xmin.append(xcenter - w / 4.0)
        ymin.append(ycenter - h / 4.0)

        xmax.append(xcenter + w / 4.0)
        ymax.append(ycenter + h / 4.0)

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
tracks = []

tracks.append([100, 100])
tracks.append([200, 200])
tracks.append([250, 250])
tracks.append([300, 250])

# main loop
prev_box = []
while(cap.isOpened()):
    print "frame:", frame_idx
    ret, np_img = cap.read()
    resized_img = cv2.resize(np_img, (448, 448))
    resized_img = cv2.resize(np_img, (448, 448))
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
    boxes, index_suppress = nonmax.non_max_suppression(boxes, 0.5)

    classes = [classes_name[i] for i in class_num]

    print "number of boxes:", len(xmin)
    print "Class:", classes
    print "Prob: ", prob
    print "boxes:", boxes
    print "index:", index_suppress

    prev_box = boxes


    if frame_idx > 0 and len(prev_box) != 0:
        # tracking:
        tracks = []
        for n in range(len(prev_box)):
            tracks.append([(boxes[n,0]+boxes[n,2])/2,(boxes[n,1]+boxes[n,3])/2 ])

        img0, img1 = prev_gray, frame_gray

        p0 = np.float32([tr[:] for tr in tracks]).reshape(-1, 1, 2)

        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []

        # draw tracking results
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            tr.append((x, y))
            if x > 400:
                continue

            new_tracks.append([x, y])
            # cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)
            cv2.rectangle(vis, (int(x - 10), int(y - 10)), (int(x + 10), int(y + 10)), (0, 255, 255))
            # cv2.putText(vis, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))

    prev_gray = frame_gray

    # draw:
    for i in range(len(boxes)):
        class_name = classes_name[class_num[i]]
        cv2.rectangle(vis, (int(boxes[i,0]), int(
            boxes[i,1])), (int(boxes[i,2]), int(boxes[i,3])), (0, 0, 255))
    #    cv2.putText(resized_img, class_name[i], (int(xmin[i]), int(ymin[i])), 2, 1.5, (0, 0, 255))
    cv2.imshow('cars', vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx = frame_idx + 1

cap.release()
cv2.destroyAllWindows()

sess.close()
