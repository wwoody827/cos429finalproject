import numpy as np
import cv2
print cv2.__version__
cap = cv2.VideoCapture('cars.avi')

print cap.isOpened()
