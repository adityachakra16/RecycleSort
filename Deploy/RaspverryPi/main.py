from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
import picamera

from PIL import Image
from tflite_runtime.interpreter import Interpreter
import stepper as stepper


def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

def turnMotor(label):
    if label == "MetalCans":

        print ("forward")
        for i in range(25):
            stepper.forwardStep()
        print ("backward")
        time.sleep(2)

        for i in range(25):
            stepper.backwardStep()

    elif label == "PlasticBottles":
        print ("backward")
        for i in range(25):
            stepper.backwardStep()
        time.sleep(2)
        print ("forward")
        for i in range(25):
            stepper.forwardStep()

#TODO: Add interrupt to start camera instead of running camera all the time
def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', default="assets/recyclesort_classifier_mobilenetv2_epoch=2_v7.tflite")
  parser.add_argument(
      '--labels', help='File path of labels file.', default="assets/recyclesort_labels.txt")
  args = parser.parse_args()

  labels = load_labels(args.labels)

  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  #Setup Raspberry Pi outputs for Stepper Motor
  stepper.setup()

  with picamera.PiCamera(resolution=(640, 480), framerate=30) as camera:
    camera.start_preview()
    try:
      stream = io.BytesIO()
      for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
        stream.seek(0)
        image = Image.open(stream).convert('RGB').resize((width, height),
                                                         Image.ANTIALIAS)
        start_time = time.time()
        results = classify_image(interpreter, image)
        elapsed_ms = (time.time() - start_time) * 1000
        label_id, prob = results[0]

        print (labels[label_id], ":", prob)
        turnMotor(labels[label_id])
        stream.seek(0)
        stream.truncate()
        camera.annotate_text = '%s %.2f\n%.1fms' % (labels[label_id], prob,
                                                    elapsed_ms)
    finally:
      camera.stop_preview()
      pass

if __name__ == '__main__':
  main()
