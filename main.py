#
#
#
import numpy as np
import onnx


def convertOnnxmodel_dynamic_shape(model_path):
    model = onnx.load(model_path)
    dynamic_out = model_path.split('.')[0] + "_dynamic.onnx"
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = '?'
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '?'
    # output
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = '?'
    model.graph.output[1].type.tensor_type.shape.dim[0].dim_param = '?'
    model.graph.output[2].type.tensor_type.shape.dim[0].dim_param = '?'




    onnx.save(model, dynamic_out)

if __name__=='__main__':
    model_path = "/home/ai/EkinStash/ai-inference/FaceRecognition_Project/FaceModels/onnxmodel/retinaface_mb025.onnx"
    convertOnnxmodel_dynamic_shape(model_path)
