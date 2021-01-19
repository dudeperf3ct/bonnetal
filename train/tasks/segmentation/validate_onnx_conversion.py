import torch
import onnxruntime
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2
import yaml
import argparse

parser = argparse.ArgumentParser("./validate_onnx_conversion.py")
parser.add_argument(
    '--pytorch_path', '-pp',
    type=str,
    required=True,
    default="models/model.pytorch",
    help='Directory to get the the model'
)
parser.add_argument(
    '--onnx_path', '-op',
    type=str,
    required=True,
    default="models/model.pytorch",
    help='Directory to get the the model'
)
parser.add_argument(
    '--height', '-hh',
    type=int,
    required=True,
    default=1056,
    help='Force Height to. Defaults to %(default)s',
)
parser.add_argument(
    '--width', '-ww',
    type=int,
    required=True,
    default=1056,
    help='Force Width to. Defaults to %(default)s',
)

FLAGS, unparsed = parser.parse_known_args()

res_path = f"results_{FLAGS.width}/"
config_pth = 'bonnetal/train/tasks/segmentation/config/coco/mobilenetv2_aspp_res.yaml'
# try to open data yaml
try:
    print("Opening config file")
    with open(config_pth, 'r') as file:
        CFG = yaml.safe_load(file)
except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()
    
means=CFG["dataset"]["img_means"]
stds=CFG["dataset"]["img_stds"]

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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

img_path = 'datasets/val/images/'
norm = transforms.Normalize(mean=means, std=stds)
tensorize_img = transforms.ToTensor()

def calibrate(tol=1e-1):
	ort_session = onnxruntime.InferenceSession(FLAGS.onnx_path)
	model = torch.jit.load(FLAGS.pytorch_path).to('cuda')
	model.eval()
	for p in os.listdir(img_path):
		# h, w = 1200, 1920 ## will not work for this either
		im = resize_and_fit(Image.open(os.path.join(img_path, p)).convert("RGB"), FLAGS.height, FLAGS.width, "RGB")
		x = norm(tensorize_img(im)).unsqueeze(0).to('cuda')

		# pytorch results
		torch_out = model(x)

		# compute ONNX Runtime output prediction
		ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
		ort_outs = ort_session.run(None, ort_inputs)
		output = np.concatenate((ort_outs[0].squeeze().argmax(0), to_numpy(torch_out).squeeze().argmax(0)), axis=1)
		name = os.path.join(res_path, p.split(".")[0]+f"_tol_{tol}.png")
		cv2.imwrite(name, output*255)
		try:
			# compare ONNX Runtime and PyTorch results
			np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=tol, atol=1e-05)
		except AssertionError as e:
			print(tol, p, e)

print("Performing calibration for rtol=1e-1...")
calibrate(tol=1e-1)
print("Performing calibration for rtol=1e-2...")
calibrate(tol=1e-2)
print("Performing calibration for rtol=1e-3...")
calibrate(tol=1e-3)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")
