# faceRecognitionOnnxruntime
face recognition using onnxruntime platfpform

language: c++
requirements:
  1- opencv
  2- onnxruntime
  opencv and onnxruntime should be linked to the project using cmake (modify cmake to make it work)
  
goals: image recognition

include:
face detector and feature extractore models. 

after face detection, flattened input is cropped and alined at the same time to remove other extra operations
of converting image to vector and vice versa. The alignment and flattening part is implemented from scratch.

  
