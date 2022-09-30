# SSD MobileNet

The mobilenet-ssd model is a Single-Shot multibox Detection (SSD) network intended to perform object detection.
Model standalone is saved in frozen_inference_graph.pb file. Program loads graph and makes detections using tensorflow and opencv libraries.

Main reason for use of this model was to set up a model for object detection on Raspberry Pi 3 with camera.
1 GB RAM and 1.2 GHz processor doesn't allow big models to run in real-time, so a smaller model was needed.
Mobienet-ssd works quite well, with detections in 1-2 FPS, so real-time detection condition is met.

Program is divided into 2 parts:

## detect_pil.py
This part I used mostly on my computer with Windows for debugging and fixing problems. It uses PIL library
instead of opencv for image processing (didn't have opencv on computer and installation caused problem so I used easier solution).
Usable only with single images.

## detect_cam.py
This is the main program. Used for detections with camera. As a bonus I included function for sending messages to user's phone.
For that reason it uses Twilio library. However, it was added for personal needs only. Twilio requires account on [their site](https://www.twilio.com/docs/sms).
Credentials for an account are left empty as global variables for one to fill.
