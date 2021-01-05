#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import time
from PIL import Image
import numpy as np
import __init__ as booger
import collections
import copy
import os

from backbones.config import *
from tasks.segmentation.modules.head import *
from tasks.segmentation.modules.segmentator import *
import onnx


class TraceSaver():
  def __init__(self, path, new_path, force_img_prop=(None, None)):
    # parameters
    self.path = path
    self.new_path = new_path
    custom_w, custom_h = 1920, 1200

    # config from path
    try:
      yaml_path = self.path + "/cfg.yaml"
      print("Opening config file %s" % yaml_path)
      self.CFG = yaml.safe_load(open(yaml_path, 'r'))
    except Exception as e:
      print(e)
      print("Error opening cfg.yaml file from trained model.")
      quit()

    # if force img prop is a tuple with 2 elements, force image props
    if force_img_prop[0] is not None or force_img_prop[1] is not None:
      if force_img_prop[0] is not None:
        self.CFG["dataset"]["img_prop"]["height"] = force_img_prop[0]
      if force_img_prop[1] is not None:
        self.CFG["dataset"]["img_prop"]["width"] = force_img_prop[1]
      print("WARNING: FORCING IMAGE PROPERTIES TO")
      print(self.CFG["dataset"]["img_prop"])

    # get the data
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/segmentation/dataset/' +
                                   self.CFG["dataset"]["name"] + '/parser.py')
    self.parser = parserModule.Parser(img_prop=self.CFG["dataset"]["img_prop"],
                                      img_means=self.CFG["dataset"]["img_means"],
                                      img_stds=self.CFG["dataset"]["img_stds"],
                                      classes=self.CFG["dataset"]["labels"],
                                      train=False)
    self.data_h, self.data_w, self.data_d = self.parser.get_img_size()

    # get architecture and build backbone (with pretrained weights)
    self.bbone_cfg = BackboneConfig(name=self.CFG["backbone"]["name"],
                                    os=self.CFG["backbone"]["OS"],
                                    h=self.data_h,
                                    w=self.data_w,
                                    d=self.data_d,
                                    dropout=self.CFG["backbone"]["dropout"],
                                    bn_d=self.CFG["backbone"]["bn_d"],
                                    extra=self.CFG["backbone"]["extra"])

    self.decoder_cfg = DecoderConfig(name=self.CFG["decoder"]["name"],
                                     dropout=self.CFG["decoder"]["dropout"],
                                     bn_d=self.CFG["decoder"]["bn_d"],
                                     extra=self.CFG["decoder"]["extra"])

    self.head_cfg = HeadConfig(n_class=self.parser.get_n_classes(),
                               dropout=self.CFG["head"]["dropout"])

    # concatenate the encoder and the head
    with torch.no_grad():
      # load the best trained weights
      # best weights are stored in _train
      ##https://github.com/dudeperf3ct/bonnetal/blob/0a921c3be379520e313883faa77e0a88c34df4ce/train/tasks/segmentation/modules/trainer.py#L319
      self.model = Segmentator(self.bbone_cfg,
                               self.decoder_cfg,
                               self.head_cfg,
                               self.path,
                               "_train",
                               strict=True)

    # CUDA speedup?
    if torch.cuda.is_available():
      cudnn.fastest = True
      cudnn.benchmark = True
      self.model = self.model.cuda()

    # don't train
    self.model.eval()
    for w in self.model.parameters():
      w.requires_grad = False

    # print number of parameters and the ones requiring gradients
    weights_total = sum(p.numel()
                        for p in self.model.parameters())
    weights_grad = sum(p.numel()
                       for p in self.model.parameters() if p.requires_grad)
    print("Total number of parameters: ", weights_total)
    print("Total number of parameters requires_grad: ", weights_grad)

    # profiler based saver, so create a dummy input to infer
    print("Creating dummy input to profile")
    self.dummy_input = torch.randn(1, self.CFG["dataset"]["img_prop"]["depth"],
                                   self.CFG["dataset"]["img_prop"]["height"],
                                   self.CFG["dataset"]["img_prop"]["width"])

    self.dummy_input1 = torch.randn(1, self.CFG["dataset"]["img_prop"]["depth"],
                                   custom_h,
                                   custom_w)
    # gpu?
    if torch.cuda.is_available():
      self.dummy_input = self.dummy_input.cuda()
      self.dummy_input1 = self.dummy_input1.cuda()

  def export_config(self):
    # save the config file in the log folder
    try:
      new_yaml_path = self.new_path + "/cfg.yaml"
      print("Saving config file %s" % new_yaml_path)
      with open(new_yaml_path, 'w') as outfile:
        yaml.dump(self.CFG, outfile, default_flow_style=False)
    except Exception as e:
      print(e)
      print("Error saving cfg.yaml in new model dir ", new_yaml_path)
      quit()

  def export_ONNX(self):
    # convert to ONNX traced model

    # create profile
    onnx_path = os.path.join(self.new_path, "model.onnx")
    with torch.no_grad():
      print("Profiling model")
      print("saving model in ", onnx_path)
      torch.onnx.export(self.model, self.dummy_input, onnx_path)

    # check that it worked
    print("Checking that it all worked out")
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)

    # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(model_onnx.graph))
    
  def export_dynamic_ONNX(self):
    # convert to a dynamic ONNX traced model
    ## NOTE: Dynamic width/height may not achieve the expected performance improvement 
    ## with some backend such as TensorRT though.
    input_names=['input']
    output_names=['output']
    dynamic_axes= {'input':{0:'batch_size' , 2:'width', 3:'height'}, 
                   'output':{0:'batch_size' , 2:'width', 3:'height'}}
    # create profile
    onnx_path = os.path.join(self.new_path, "model_dynamic.onnx")
    with torch.no_grad():
      print("Profiling model")
      print("saving model in ", onnx_path)
      torch.onnx.export(self.model, self.dummy_input1, onnx_path, 
                        input_names, output_names, dynamic_axes)

    # check that it worked
    print("Checking that it all worked out")
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)

    # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(model_onnx.graph))

  def export_pytorch(self):
    # convert to Pytorch traced model

    # create profile
    pytorch_path = os.path.join(self.new_path, "model.pytorch")
    with torch.no_grad():
      print("Profiling model")
      pytorch_model = torch.jit.trace(self.model, self.dummy_input)
      print("saving model in ", pytorch_path)
      pytorch_model.save(pytorch_path)

  def export(self):
    """Export config file, ONNX model, and Pytorch traced model to log directory
    """
    self.export_config()
    self.export_ONNX()
    self.export_pytorch()
    self.export_dynamic_ONNX()
