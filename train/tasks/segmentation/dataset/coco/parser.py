# This file is covered by the LICENSE file in the root of this project.

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import random
import torchvision.transforms.functional as TF
import cv2
import albumentations


'''
means(rgb):  [0.38696613 0.4110715  0.41329755]
stds(rgb):  [0.05977108 0.05584759 0.06669568

Num of pixels:  4433551360
Frequency:  [0.96726937 0.03226973]
********************************************************************************
Log strategy
 Weights:  [ 1.45610953 19.62728561]
Linear strategy
 Weights:  [0.03273063 0.96773027]
Squared strategy
 Weights:  [0.00107129 0.93650187]
1/w strategy
 Weights:  [ 1.03383816 30.98878247]

'''
IMG_EXT = ['.jpg']

LBL_EXT = ['.png']
SCALES = [1.0]

# strong augmentations
def strong_augemntations(p=0.5):
    return albumentations.Compose([
        albumentations.RandomRotate90(),
        albumentations.Flip(),
        albumentations.Transpose(),
        albumentations.OneOf([
            albumentations.IAAAdditiveGaussianNoise(),
            albumentations.GaussNoise(),
        ], p=0.2),
        albumentations.OneOf([
            albumentations.MotionBlur(p=0.2),
            albumentations.MedianBlur(blur_limit=3, p=0.1),
            albumentations.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        albumentations.OneOf([
            albumentations.OpticalDistortion(p=0.3),
            albumentations.GridDistortion(p=0.1),
            albumentations.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        albumentations.OneOf([
            albumentations.CLAHE(clip_limit=2),
            albumentations.IAASharpen(),
            albumentations.IAAEmboss(),
            albumentations.RandomBrightnessContrast(),
        ], p=0.3),
        albumentations.HueSaturationValue(p=0.3),
    ], p=p)


class ToLabel:
  def __call__(self, label):
    label = np.array(label)
    return torch.from_numpy(label).long()


def load_image(file):
  return Image.open(file)


def load_label(file):
  return Image.open(file)


def is_image(filename):
  return any(filename.endswith(ext) for ext in IMG_EXT)


def is_label(filename):
  return any(filename.endswith(ext) for ext in LBL_EXT)


def resize_and_fit(img, new_h, new_w, img_type):
  # check img_type
  assert(img_type is "RGB" or img_type is "L")

  # get current size
  w, h = img.size

  # generate new img
  out_img = Image.new(img_type, (new_w, new_h))

  # now do size magic
  curr_asp_ratio = h / w
  new_asp_ratio = new_h / new_w

  # do resizing according to aspect ratio
  if curr_asp_ratio > new_asp_ratio:
    # fit h to h
    new_tmp_h = new_h
    new_tmp_w = int(w * new_h / h)
  else:
    # fit w to w
    new_tmp_w = new_w
    new_tmp_h = int(h * new_w / w)

  # resize the original image
  if img_type is "RGB":
    tmp_img = img.resize((new_tmp_w, new_tmp_h), Image.BILINEAR)
  else:
    tmp_img = img.resize((new_tmp_w, new_tmp_h), Image.NEAREST)

  # put in padded image
  out_img.paste(tmp_img, (int((new_w-new_tmp_w)//2),
                          int((new_h-new_tmp_h)//2)))

  return out_img


class MS_COCO(Dataset):
  def __init__(self, root, subset, h, w, means, stds, crop_h=None, crop_w=None, strong_aug=False):
    self.images_root = os.path.join(root, subset, "images")
    self.labels_root = os.path.join(root, subset, subset+"_remap")

    self.subset = subset
    assert self.subset == 'train' or self.subset == 'val'

    self.w = w
    self.h = h
    self.means = means
    self.stds = stds
    self.strong_augment = strong_aug
    self.aug = strong_augemntations(p=0.5)

    if self.subset == 'train':
      self.crop_h = crop_h
      self.crop_w = crop_w

      # check that parameters make sense
      assert(self.crop_h <= self.h)
      assert(self.crop_w <= self.w)

      self.resize_crop_img = transforms.Resize((self.crop_h, self.crop_w),
                                               Image.BILINEAR)
      self.resize_crop_lbl = transforms.Resize((self.crop_h, self.crop_w),
                                               Image.NEAREST)

    print("Images from: ", self.images_root)
    print("Labels from: ", self.labels_root)

    self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
    self.filenames.sort()

    self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
    self.filenamesGt.sort()

    assert len(self.filenames) == len(self.filenamesGt)

    # transformations for images
    self.jitter = transforms.ColorJitter(brightness=0.05,
                                         contrast=0.05,
                                         saturation=0.05,
                                         hue=0.05)
    self.h_flip = TF.hflip
    self.crop_param = transforms.RandomCrop.get_params
    self.crop = TF.crop

    # transformations for tensors
    self.norm = transforms.Normalize(mean=self.means, std=self.stds)
    self.tensorize_img = transforms.ToTensor()
    self.tensorize_lbl = ToLabel()

  def __getitem__(self, index):
    filename = self.filenames[index]
    filenameGt = self.filenamesGt[index]

    with open(filename, 'rb') as f:
      image = load_image(f).convert('RGB')
    with open(filenameGt, 'rb') as f:
      label = load_label(f).convert('L')

    # resize (resizing is different if we are in train or valid mode)
    # generate resizer
    if self.subset == 'train':
      new_h = self.crop_h
      new_w = self.crop_w
    else:
      new_h = self.h
      new_w = self.w
    image = resize_and_fit(image, new_h, new_w, "RGB")
    label = resize_and_fit(label, new_h, new_w, "L")

    # augment data and tensorize
    if self.subset == 'train':
      # crop randomly sized patches
      scale = SCALES[random.randrange(len(SCALES))]
      size = (int(self.crop_h * scale), int(self.crop_w * scale))
      i, j, h, w = self.crop_param(image, output_size=size)
      image = self.resize_crop_img(self.crop(image, i, j, h, w))
      label = self.resize_crop_lbl(self.crop(label, i, j, h, w))

      # apply strong augmentations only for training set
      if self.strong_augment:
        data = {"image": np.array(image), "label": np.array(label)}
        augmented = self.aug(**data)
        image, label = augmented["image"], augmented["label"]

      # basic augmentation if strong augmentation not used
      else:
        # flip
        if random.random() > 0.5:
          image = self.h_flip(image)
          label = self.h_flip(label)

        # jitter
        if random.random() > 0.5:
          image = self.jitter(image)

      # show (set workers = 0)
      # cv2.imshow("train_img", np.array(image)[:, :, ::-1])
      # cv2.imshow("train_lbl", LUT[np.array(label)].astype(np.float32) / 21.0)
      # cv2.waitKey(0)

    # if self.subset == 'val':
    #   show (set workers = 0)
    #   cv2.imshow("valid_img", np.array(image)[:, :, ::-1])
    #   cv2.waitKey(0)

    # tensorize
    image = self.tensorize_img(image)
    label = self.tensorize_lbl(label)

    # normalize
    image = self.norm(image)

    return image, label

  def __len__(self):
    return len(self.filenames)


class Parser():
  # standard conv, BN, relu
  def __init__(self, img_prop, img_means, img_stds, classes, train, location=None, batch_size=None, crop_prop=None, workers=2, strong_aug=False):
    super(Parser, self).__init__()

    self.img_prop = img_prop
    self.img_means = img_means
    self.img_stds = img_stds
    self.classes = classes
    self.train = train
    self.strong_aug = strong_aug

    if self.train:
      # if I am training, get the dataset
      self.location = location
      self.batch_size = batch_size
      self.crop_prop = crop_prop
      self.workers = workers

      # Data loading code
      self.train_dataset = MS_COCO(root=self.location,
                                   subset='train',
                                   h=self.img_prop["height"],
                                   w=self.img_prop["width"],
                                   means=self.img_means,
                                   stds=self.img_stds,
                                   crop_h=self.crop_prop["height"],
                                   crop_w=self.crop_prop["width"],
                                   strong_aug=self.strong_aug)

      self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=True,
                                                     num_workers=self.workers,
                                                     pin_memory=True,
                                                     drop_last=True)
      assert len(self.trainloader) > 0
      self.trainiter = iter(self.trainloader)

      # calculate validation batch from train batch and image sizes
      factor_val_over_train = float(self.img_prop["height"] * self.img_prop["width"]) / float(
          self.crop_prop["height"] * self.crop_prop["width"])
      self.val_batch_size = max(
          1, int(self.batch_size / factor_val_over_train))

      # if gpus are available make val_batch_size at least the number of gpus
      if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        self.val_batch_size = max(
            self.val_batch_size, torch.cuda.device_count())

      print("Inference batch size: ", self.val_batch_size)

      self.valid_dataset = MS_COCO(root=self.location,
                                   subset='val',
                                   h=self.img_prop["height"],
                                   w=self.img_prop["width"],
                                   means=self.img_means,
                                   stds=self.img_stds)

      self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                     batch_size=self.val_batch_size,
                                                     shuffle=False,
                                                     num_workers=self.workers,
                                                     pin_memory=True,
                                                     drop_last=True)
      assert len(self.validloader) > 0
      self.validiter = iter(self.validloader)

  def get_train_batch(self):
    images, labels = self.trainiter.next()
    return images, labels

  def get_train_set(self):
    return self.trainloader

  def get_valid_batch(self):
    images, labels = self.validiter.next()
    return images, labels

  def get_valid_set(self):
    return self.validloader

  def get_train_size(self):
    return len(self.trainloader)

  def get_valid_size(self):
    return len(self.validloader)

  def get_img_size(self):
    h = self.img_prop["height"]
    w = self.img_prop["width"]
    d = self.img_prop["depth"]
    return h, w, d

  def get_n_classes(self):
    return len(self.classes)

  def get_class_string(self, idx):
    return self.classes[idx]

  def get_means_stds(self):
    return self.img_means, self.img_stds
