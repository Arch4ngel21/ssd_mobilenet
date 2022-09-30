import time
import datetime

import numpy as np
import tensorflow as tf
from typing import List
import cv2
import twilio.rest
from twilio.rest import Client


motion_detected = False
ACCOUNT_SID = ""
AUTH_TOKEN = ""
ACCOUNT_NUMBER = ""
RECV_NUMBER = ""


def load_network():
    GRAPH_PB_PATH = './frozen_inference_graph.pb'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(GRAPH_PB_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph


def load_labels(file_path: str) -> List:
    labels = ["X"]
    with open(file_path, "r") as label_file:
        lines = label_file.readlines()
        for line in lines:
            if line[1] == ' ':
                line = line[2:-1]
            else:
                line = line[3:-1]
            labels.append(line)

    return labels


def init_twilio_client():
    global ACCOUNT_SID, AUTH_TOKEN
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    return client


def send_alert(client: twilio.rest.Client, object_type: str):
    global motion_detected, ACCOUNT_NUMBER
    message = client.messages.create(
        body=f"Time: {str(datetime.datetime.now())} - motion detected, type: {object_type}",
        from_=ACCOUNT_NUMBER,
        to=RECV_NUMBER
    )
    motion_detected = True
    print("Message send -", message.sid)


if __name__ == '__main__':
    NUM_CLASSES = 90
    ANIMALS = {"dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear"}

    mobilenet = load_network()
    print("##### Network loaded #####")
    labels = np.array(load_labels('coco_labels.txt'))
    print("##### Labels loaded #####")
    client = init_twilio_client()
    print("##### Client loaded #####")

    cap = cv2.VideoCapture(0)

    with mobilenet.as_default():
        with tf.compat.v1.Session(graph=mobilenet) as sess:
            while True:
                ret, frame = cap.read()

                img_org = frame
                img_as_array = np.asarray(img_org)

                img_expanded = np.expand_dims(img_org, axis=0)
                image_tensor = mobilenet.get_tensor_by_name('image_tensor:0')
                boxes = mobilenet.get_tensor_by_name('detection_boxes:0')
                scores = mobilenet.get_tensor_by_name('detection_scores:0')
                classes = mobilenet.get_tensor_by_name('detection_classes:0')
                num_detections = mobilenet.get_tensor_by_name('num_detections:0')

                start = time.time()

                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: img_expanded})

                end = time.time() - start
                print(f"### FPS: {1 / end}")

                boxes = boxes.reshape((-1, boxes.shape[2]))
                scores = scores.flatten()
                classes = classes.flatten()

                # print("----- Boxes -----")
                # print(boxes)
                # print("----- Scores -----")
                # print(scores)
                # print("----- Classes -----")
                # print(classes)
                # print("----- Num detections -----")
                # print(num_detections)
                try:
                    for detection in range(int(num_detections[0])):
                        if not motion_detected and str(labels[int(classes[detection])]) == "person":    # in ANIMALS:
                            send_alert(client, str(labels[int(classes[detection])]))

                        x0 = int(boxes[detection][0] * img_org.shape[0])
                        y0 = int(boxes[detection][1] * img_org.shape[1])
                        x1 = int(boxes[detection][2] * img_org.shape[0])
                        y1 = int(boxes[detection][3] * img_org.shape[1])

                        # print('coordinates: {} {}. class: {}. confidence: {:.2f}'.format((x0, y0), (x1, y1), str(labels[int(classes[detection])]), scores[detection]))

                        cv2.rectangle(img_org, (x0, y0), (x1, y1), (0, 255, 0), 2)
                        cv2.putText(img_org, '{}: {:.2f}'.format(str(labels[int(classes[detection])]), scores[detection]),
                                    (x0, y0 - 5),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                except IndexError:
                    print("IndexError detected!")
                    continue

                cv2.imshow("Stream", img_org)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break

    exit(0)
