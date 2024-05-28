from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from sort import *
from fns import *
 
cap = cv2.VideoCapture("../vid/cars.mp4")  # For Video
model = YOLO("../Yolo-weights/yolov8n.pt")
mask = cv2.imread("../images/mask-car.png")

#Tracking
tracker = Sort(max_age=50, min_hits=2, iou_threshold=0.3)
unique_cars = set()
limits = [400,297,673,297]
while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)
    results = model(imgRegion, stream=True)
    detections=np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])

            className = classNames[cls]
            if className in Trackable and conf>0.3:
                # cvzone.cornerRect(img, (x1, y1, w, h))
                # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))
    
    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0],limits[1]), (limits[2],limits[3]),(0,0,255))
    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        print(result)
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1
        cx,cy = x1 + w//2, y1 + h//2
        cv2.circle(img, (cx,cy),5,(255,0,255),cv2.FILLED)
        if limits[0]<cx<limits[2] and limits[1]-15<cy<limits[3]+15:
            unique_cars.add(id)
        
        cvzone.cornerRect(img, (x1, y1, w, h))
        cvzone.putTextRect(img, f'{classNames[cls]} {id}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
        cvzone.putTextRect(img, f'{len(unique_cars)}', (50,50), scale=1, thickness=1)



    cv2.imshow("Image", img)
    cv2.waitKey(1)