import time

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from typing import List
import cv2
from PIL import Image, ImageDraw, ImageFont


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


if __name__ == '__main__':
    NUM_CLASSES = 90
    font = ImageFont.truetype("arial.ttf", 25)
    mobilenet = load_network()
    labels = np.array(load_labels('coco_labels.txt'))

    with mobilenet.as_default():
        with tf.compat.v1.Session(graph=mobilenet) as sess:
            img_org = Image.open('people.jpg')

            img_resized = img_org.resize((640, 480))
            img_as_array = np.asarray(img_resized)

            draw = ImageDraw.Draw(img_org)
            img_expanded = np.expand_dims(img_resized, axis=0)
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
            print(f"Found {int(num_detections[0])} objects in {end} s.")

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

            for detection in range(int(num_detections[0])):
                x0 = int(boxes[detection][0] * img_org.size[0])
                y0 = int(boxes[detection][1] * img_org.size[1])
                x1 = int(boxes[detection][2] * img_org.size[0])
                y1 = int(boxes[detection][3] * img_org.size[1])
                draw.rectangle([(x0, y0), (x1, y1)], fill=None, outline=(0, 255, 0))
                draw.text((x0, y0), str(labels[int(classes[detection])]), fill=(0, 255, 0), font=font)
                draw.text((x0, y0 + 35), str(scores[detection]), fill=(0, 255, 0), font=font)

            img_org.show()

    exit(0)

    # predictor = ObjectDetector()
    # camera = PiCamera()
    # camera.resolution(640, 480)
    # camera.framerate(32)
    # rawCapture = PiRGBArray(camera, size=(640, 480))

    # time.sleep(0.1)

    """for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        result = predictor.detect(image)

        for obj in result:
            logger.info('coordinates: {} {}. class: {}. confidence: {:.2f}'.format(obj[0], obj[1], obj[3], obj[2]))

            cv2.rectangle(image, obj[0], obj[1], (0, 255, 0), 2)
            cv2.putText(image, '{}: {:.2f}'.format(obj[3], obj[2]), (obj[0][0], obj[0][1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        cv2.imshow("Stream", image)
        key = cv2.waitKey(1) & 0xFF"""
