#
import numpy as np
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet

from onnx import checker
import onnx


model_path = "./models/mxnetmodel/"

symb = "2d106det-symbol.json"
param = "2d106det-0000.params"

if __name__ == '__main__':

    input_shape = (1, 3, 192, 192)
    onnx_file = "MobileNet_2d106det_2.onnx"

    converted_model_path = onnx_mxnet.export_model( model_path + symb, model_path + param, [input_shape], np.float32, model_path + onnx_file)


    # check model
    # Load onnx model
    model_proto = onnx.load_model(converted_model_path)

    # Check if converted ONNX protobuf is valid
    checker.check_graph(model_proto.graph)
