#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import yaml
import os
import cv2
import numpy as np
import json
import shutil
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

def annToRLE(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann['segmentation']
    return rle

def annToMask(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m

LBL_EXT = ['.png']
IMG_EXT = ['.jpg', ]
SUBSETS = ['train', 'val']


def is_label(filename):
  return any(filename.endswith(ext) for ext in LBL_EXT)


def is_image(filename):
  return any(filename.endswith(ext) for ext in IMG_EXT)


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./generate_gt.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      default=None,
      help='Directory to get the dataset from.'
  )
  FLAGS, unparsed = parser.parse_known_args()

  # want map of shape (instance count, height, width)
  per_instance_map = False

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("dataset dir", FLAGS.dataset)
  print("----------\n")
  # print("Commit hash (training version): ", str(
  #     subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
  # print("----------\n")

  # try to open data yaml
  try:
    print("Opening config file")
    with open('cfg.yaml', 'r') as file:
      CFG = yaml.safe_load(file)
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  # make lut for remap
  max_val = 0
  for key, val in CFG["int_mapping"].items():
    if val > max_val:
      max_val = val
  assert(max_val < 256)
  LUT = np.zeros(max_val + 1, dtype=np.uint8)
  for key, val in CFG["int_mapping"].items():
    LUT[val] = key  # inverse mapping for cross entropy
  print ("LUT:", LUT)

  for subset in SUBSETS:
    SUBSET_JSON = os.path.join(FLAGS.dataset, subset, subset + ".json")  #json label
    SUBSET_DIR = os.path.join(FLAGS.dataset, subset, "images")  #images
    coco = COCO(SUBSET_JSON)
    print("Getting labels from: ")
    print(SUBSET_JSON)
    print(SUBSET_DIR)

    # now make the remap directory
    REMAP_DIR = os.path.join(FLAGS.dataset, subset, subset + "_remap") #new ground truth
    print("putting labels in")
    print(REMAP_DIR)
    # create folder
    try:
      if os.path.isdir(REMAP_DIR):
        shutil.rmtree(REMAP_DIR)
      os.makedirs(REMAP_DIR)
    except Exception as e:
      print(e)
      print("Error creating", REMAP_DIR, ". Check permissions!")
      quit()

    # open json file
    with open(SUBSET_JSON) as file:
      subset_json = json.load(file)
      subset_labels = subset_json["images"]
      for label in subset_labels:
        label_file = os.path.join(SUBSET_DIR, label["file_name"])
        # open file BGR
        print("Getting instance image from ", label_file)
        # lbl = cv2.imread(label_file, cv2.IMREAD_COLOR)
        # B, G, R = cv2.split(lbl)
        # B = B.astype(np.uint32)
        # G = G.astype(np.uint32)
        # R = R.astype(np.uint32)

        # show for debugging purposes
        # cv2.imshow("label", lbl)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # output shape = (height, width)  
        # where each number corresponding to category(instance) that mask belongs to   
        mask = np.zeros(( label["height"], label["width"]))
        # make remap from instances in label
        annotation = coco.loadAnns(coco.getAnnIds(imgIds=[label["id"]], iscrowd=None))
        m = []
        for anno in annotation:
          #binary mask for one instance in current label
          seg = annToMask(anno, label["height"], label["width"]).astype(np.uint8)
          # Some objects are so small that they're less than 1 pixel area
          # and end up rounded out. Skip those objects.
          if seg.max() < 1:
            continue
          if per_instance_map:
            m.append(seg)  
          # (height, width)
          # # hacky way to bring all the values to 1 ?? not sure if it will work ??     
          else:
            mask = np.logical_or(mask, m)

        # output shape = (instance count, height, width)
        # where each instance count will contain binary mask
        if per_instance_map:
          mask = np.array(m, axis=-1).astype(np.uint8)
          # (height, width, instance count)
          mask = np.transpose(mask, (1, 2, 0))
        # output shape = (height, width)  
        # where each number corresponding to category(instance) that mask belongs to 
        # else:
        #   # (height, width)
        #   # hacky way to bring all the values to 1 ?? not sure if it will work ??
        #   mask = np.sum(np.array(m), axis=0).astype(np.uint8)
        #   row, col = np.where(mask>1)
        #   mask[np.ix_(row,col)] = 1

        # # pass to my class defition
        # not required as our mapping has only 0 and 1
        # mask = LUT[mask]
        print (mask.shape)
        print (np.unique(mask, return_counts=True))

        # show
        # cv2.imshow("inst_remapped", inst_remapped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # now save in remapped directory
        path_remapped = os.path.join(REMAP_DIR, label["file_name"].replace("jpg", "png"))
        cv2.imwrite(path_remapped, mask)
        # break