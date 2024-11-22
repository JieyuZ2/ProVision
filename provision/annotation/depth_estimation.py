import os.path

import cv2
import numpy as np
import requests
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import pipeline


def save_depth_res(
		image_size,
		depth,
		save_path: str,
		grayscale: bool = False,
):
	depth = np.array(depth)
	depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
	depth = depth.astype(np.uint8)

	# if grayscale:
	# 	depth_img = Image.fromarray(np.repeat(depth[..., np.newaxis], 3, axis=-1))
	# else:
	# 	depth_img = Image.fromarray(cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO))

	np.save(save_path, depth)
	return save_path


def depth_estimation(
		image_list: list,
		model: str = "depth-anything/Depth-Anything-V2-Large-hf",
		device: str = "cpu",
		save_path: str = "./depth_pred_res/",
		**kwargs
):
	# load pipe
	pipe = pipeline(task="depth-estimation", model=model, device=device, **kwargs)
	if not os.path.isdir(save_path):
		os.mkdir(save_path)

	# inference
	res = []
	for image in tqdm(image_list, desc="depth estimation"):
		if "http" in image:
			image_name = image.split('/')[-1]
		else:
			image_name = os.path.basename(image)

		filename = os.path.basename(image_name)
		path = os.path.join(save_path, filename[:filename.rfind('.')] + '_depth.npy')

		if not os.path.exists(path):
			if "http" in image:
				image_obj = Image.open(requests.get(image, stream=True).raw)
			else:
				image_obj = Image.open(image)

			# inference
			depth = pipe(image_obj)['depth']
			path = save_depth_res((image_obj.height, image_obj.width),
									  depth, path, True)

		res.append({
			"depth_mask_id": os.path.basename(path)
		})

	return res
