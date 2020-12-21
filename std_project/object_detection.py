

import numpy as np
import time
import cv2
import os


class YoloObjectDetector():
   def __init__(self, image_path, yolo_path, confidence=0.5, threshold=0.3):
      self._image_path = image_path
      self._yolo_path = yolo_path
      self._confidence = confidence
      self._threshold = threshold
      self.init_yolo_variables()


   def init_yolo_variables(self):
      # load the COCO class labels our YOLO model was trained on
      labelsPath = os.path.sep.join([self._yolo_path, "coco.names"])
      self.LABELS = open(labelsPath).read().strip().split("\n")

      # initialize a list of colors to represent each possible class label
      np.random.seed(42)
      self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")

      # derive the paths to the YOLO weights and model configuration
      self._weightsPath = os.path.sep.join([self._yolo_path, "yolov3.weights"])
      self._configPath = os.path.sep.join([self._yolo_path, "yolov3.cfg"])


   def load_yolo_net(self):
      # load our YOLO object detector trained on COCO dataset (80 classes)
      print("[INFO] loading YOLO from disk...")
      net = cv2.dnn.readNetFromDarknet(self._configPath, self._weightsPath)
      return net


   def detect_objects(self):
      net = self.load_yolo_net()

      # load our input image and grab its spatial dimensions
      image = cv2.imread(self._image_path)
      (H, W) = image.shape[:2]

      layer_outputs = self.get_yolo_output(image, net)
      (boxes, confidences, classIDs, idxs) = self.get_output_coordinates(layer_outputs, H, W)
      if len(idxs) > 0:
         self.draw_boxes(idxs, boxes, confidences, classIDs, image)

      # show the output image
      cv2.imshow("Image", image)
      cv2.waitKey(0)
      return (boxes, confidences, classIDs, idxs)


   def get_yolo_output(self, image, net):
      # determine only the *output* layer names that we need from YOLO
      ln = net.getLayerNames()
      ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
      # construct a blob from the input image and then perform a forward
      # pass of the YOLO object detector, giving us our bounding boxes and
      # associated probabilities
      blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                            swapRB=True, crop=False)
      net.setInput(blob)
      start = time.time()
      layerOutputs = net.forward(ln)
      end = time.time()
      # show timing information on YOLO
      print("[INFO] YOLO took {:.6f} seconds".format(end - start))
      return layerOutputs


   def get_output_coordinates(self, layerOutputs, H, W):
      # initialize our lists of detected bounding boxes, confidences, and
      # class IDs, respectively
      boxes = []
      confidences = []
      classIDs = []

      # loop over each of the layer outputs
      for output in layerOutputs:
         # loop over each of the detections
         for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > self._confidence:
               # scale the bounding box coordinates back relative to the
               # size of the image, keeping in mind that YOLO actually
               # returns the center (x, y)-coordinates of the bounding
               # box followed by the boxes' width and height
               box = detection[0:4] * np.array([W, H, W, H])
               (centerX, centerY, width, height) = box.astype("int")
               # use the center (x, y)-coordinates to derive the top and
               # and left corner of the bounding box
               x = int(centerX - (width / 2))
               y = int(centerY - (height / 2))
               # update our list of bounding box coordinates, confidences,
               # and class IDs
               boxes.append([x, y, int(width), int(height)])
               confidences.append(float(confidence))
               classIDs.append(classID)

      # apply non-maxima suppression to suppress weak, overlapping bounding
      # boxes
      idxs = cv2.dnn.NMSBoxes(boxes, confidences, self._confidence,
                        self._threshold)

      return boxes, confidences, classIDs, idxs


   def draw_boxes(self, idxs, boxes, confidences, classIDs, image):
         # loop over the indexes we are keeping
         for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in self.COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                     0.5, color, 2)