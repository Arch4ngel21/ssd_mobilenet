# SSD MobileNet

The mobilenet-ssd model is a Single-Shot multibox Detection (SSD) network intended to perform object detection.
Model standalone is saved in frozen_inference_graph.pb file. Program loads graph and makes detections using tensorflow and opencv libraries.

Main reason for use of this model was to set up a model for object detection on Raspberry Pi 3 with camera.
1 GB RAM and 1.2 GHz processor doesn't allow big models to run in real-time, so a smaller model was needed.
Mobienet-ssd works quite well, with detections in 1-2 FPS, so real-time detection condition is met.
