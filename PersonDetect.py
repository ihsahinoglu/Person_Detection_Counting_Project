import cv2
import datetime
import imutils
# import matplotlib.pyplot as plt
import numpy as np
from centroidtracker import CentroidTracker
from itertools import combinations
import math
from non_max_suppression_fast import non_max_suppression_fast
from collections import deque

protopath = "MobileNetSSD_deploy.prototxt"
#protopath = "deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
#modelpath = "mobilenet_iter_73000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=10, maxDistance=70)


def begin():
    # Örnek videolar https://motchallenge.net/data/MOT15/ sitesinden alınmıştır.
    cap = cv2.VideoCapture('videos//3.avi')
    #W = int(cap.get(3))
    #H = int(cap.get(4))
    outputVideo = cv2.VideoWriter('videos//outputVideo3_new.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10.0, (640, 480))
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    current_person_count = 0
    total_person_count = 0
    object_id_list = []
    start_time = dict()
    passed_time = dict()
    tracking_points = deque(maxlen=10)
    #tracking_points = dict()
    tracking_line = dict()
    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (640, 480))
        frame = imutils.resize(frame, width=640)
        total_frames = total_frames + 1
        (H, W) = frame.shape[:2]

        output = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        detector.setInput(output)
        person_detections = detector.forward()  # return [id ,Class number, possibilty, startX, startY, endX, endY]

        rects = []
        for i in np.arange(0, person_detections.shape[2]):  # 0--100
            possibility = person_detections[0, 0, i, 2]

            if possibility > 0.6:           # person posibility %60
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":  # Person object id = 15
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)

        # Aynı kişi için tek diktörtgen çizilmesi için dikdörtgenler non_max_suppression_fast metoduna gönderiliyor
        rects = non_max_suppression_fast(boundingboxes, 0.3)
        center_point_dict = dict()
        objects = tracker.update(rects)   # tespit edilen objenin id numarasına göre takibi yapılıyor

        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            centerX = int((x1 + x2) / 2.0)
            centerY = int((y1 + y2) / 2.0)
            center_point_dict[objectId] = (centerX, centerY, x1, y1, x2, y2)
            #tracking_points[objectId](maxlen=10)
            tracking_points[objectId].append((centerX, centerY))
            #tracking_line[objectId] = tracking_points()
            if objectId not in object_id_list:
                object_id_list.append(objectId)
                start_time[objectId] = datetime.datetime.now()
                passed_time[objectId] = 0
            else:
                current_time = datetime.datetime.now()
                old_time = start_time[objectId]
                time_difference = current_time - old_time
                start_time[objectId] = datetime.datetime.now()
                sec = time_difference.total_seconds()
                passed_time[objectId] += sec

            id_text = "ID:{}|".format(objectId + 1)
            time_text = "{}".format(datetime.datetime.fromtimestamp(int(passed_time[objectId])).strftime('%M:%S'))
            cv2.putText(frame, id_text + time_text, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255), 1)

            if objectId not in object_id_list:
                object_id_list.append(objectId)

        red_zone_list = []
        for (id1, p1), (id2, p2) in combinations(center_point_dict.items(), 2):
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]
            distance = math.sqrt(dx * dx + dy * dy) # Belirlenen iki kişi arasındaki mesafe
            # İki kişi arasındaki mesafe toplam genişliğin 1/8 sinden küçükse kırmızı, değilse yeşil dikdörtgem çiziliyor
            if distance < W/8:
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)

        for i, box in center_point_dict.items():
            if i in red_zone_list:
                cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2)
                if len(tracking_points[objectId]) > 1:
                    cv2.line(frame, tracking_line[objectId], tracking_line[objectId - 1], (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)
                if len(tracking_points[objectId]) > 1:
                    cv2.line(frame, tracking_line[objectId], tracking_line[objectId - 1], (0, 255, 0), 2)
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        current_person_count = len(objects)
        total_person_count = len(object_id_list)

        current_txt = "CURRENT: {}".format(current_person_count)
        total_txt = "TOTAL: {}".format(total_person_count)

        cv2.putText(frame, current_txt, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.putText(frame, total_txt, (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        #frame = imutils.resize(frame, width=Wt, height=Ht)
        #frame = cv2.resize(frame, width=Wt, height=Ht)
        outputVideo.write(frame)
        cv2.imshow("Application", frame)

        # klavyeden q tuşuna basınca program kapanıyor
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


begin()
