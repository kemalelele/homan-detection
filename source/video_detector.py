import time
import numpy as np
import cv2
import imutils
import argparse
from imutils.video import VideoStream
import winsound



# model = keras.models.load_model('models/mask_mobilenet.h5')
# face_detector = load_cascade_detector()
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "table",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
IGNORE = set (["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", 
	"dog", "horse", "motorbike", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"])
COLORS = [[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47]]

net = cv2.dnn.readNetFromCaffe("models/MobileNetSSD_deploy.prototxt.txt", "models/MobileNetSSD_deploy.caffemodel")

notification = 0

def video_mask_detector():
    video = VideoStream(src=0).start()
    time.sleep(1.0)
    while True:
        # Capture frame-by-frame
        frame = video.read()

        frame = detect_mask_in_frame(frame)
        # Display the resulting frame
        # show the output frame
        cv2.imshow("Mask detector", frame)

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # cleanup
    cv2.destroyAllWindows()
    video.stop()

def detect_mask_in_frame(frame):
    while True:
        frame = imutils.resize(frame, width=900)
        global notification

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.2:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                # print(i)
                notification = 0
                global COLORS
                if i == 2:
                    color = (0,0,255)
                    print('Lebih Dari 2 orang')
                    winsound.PlaySound("warning", winsound.SND_FILENAME)
                    cv2.putText(frame, "Lebih dari 2 orang", (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    notification = 1
                     
                    COLORS = [[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,8,255],[0,8,255],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47]]
                else:
                    COLORS = [[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47],[0,255,47]]     

                if CLASSES[idx] in IGNORE:
                    continue
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        return frame

def print_notif():
    global notification
    # print(statusNotif)
    if notification == 1:
        # print(notification)
        return notification
    return notification

if __name__ == '__main__':
    video_mask_detector()
