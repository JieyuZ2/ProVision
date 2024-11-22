import os
import pickle as pkl
from typing import Dict, List
from urllib.request import urlretrieve

import cv2
import numpy as np
import requests
from PIL import Image
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix


def prepare_sam_model(path_to_checkpoint: str, version="072824"):
	if not os.path.isfile(path_to_checkpoint):
		print("Checkpoint does not exist. Downloading the model checkpoint...", flush=True)
		file_name = path_to_checkpoint.split("/")[-1]
		try:
			urlretrieve(f"https://dl.fbaipublicfiles.com/segment_anything_2/{version}/{file_name}",
						path_to_checkpoint)
			print(f"Download completed. Checkpoint saved at {path_to_checkpoint}", flush=True)
		except:
			raise ValueError("The model checkpoint is not available.")
		return


def get_image(image_path_or_url: str):
	if "http" in image_path_or_url:
		return Image.open(requests.get(image_path_or_url, stream=True).raw).convert('RGB')
	else:
		return Image.open(image_path_or_url).convert('RGB')


def save_mask_as_png(masks, file_path: str, grey_scale: bool = False):
	### save total mask
	if isinstance(masks[0], dict):
		img = np.zeros((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1]), dtype=np.uint8)
	else:
		img = np.zeros((masks[0].shape[0], masks[0].shape[1]), dtype=np.uint8)

	if grey_scale:
		i = 30
		step = 20
	else:
		i = 1  # 0 is reserved for background
		step = 1

	for ann in masks:
		if isinstance(masks[0], dict):
			m = ann['segmentation'].astype(bool)
		else:
			m = ann.astype(bool)
		img[m] = i
		i += step

	if not grey_scale:
		# Generate random colors for each class
		np.random.seed(0)  # Ensure consistent colors across different runs
		colors = np.random.randint(0, 255, size=(np.max(img) + 1, 3), dtype=np.uint8)

		# Create a transparent mask overlay
		color_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
		for label in np.unique(img):
			if label > 0:  # Skip background (label 0)
				color_image[img == label] = colors[label]

		color_image = Image.fromarray(color_image.astype(np.uint8), mode="RGB")
	else:
		color_image = Image.fromarray(np.repeat(img[..., np.newaxis], 3, axis=-1))

	parent_path = file_path[:file_path.rfind(os.path.basename(file_path))]
	if not os.path.isdir(parent_path):
		os.mkdir(parent_path)
	color_image.save(file_path)

	return file_path


def save_sep_mask(masks: List[Dict], save_path: str):
	whole_mask = []
	bboxes = []
	for mask in masks:
		whole_mask.append(csr_matrix(mask['segmentation'], dtype=bool))
		bboxes.append(mask['bbox'])

	with open(save_path, 'wb') as f:
		pkl.dump(whole_mask, f)  # list[coo_matrix]

	return save_path, bboxes


def show_points(coords, labels, ax, marker_size=375):
	pos_points = coords[labels == 1]
	neg_points = coords[labels == 0]
	ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
			   linewidth=1.25)
	ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
			   linewidth=1.25)


def visualize_obj_det(image_path, boxes, scores, labels):
	# Load image
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

	# Plot each bounding box
	for box, score, label in zip(boxes, scores, labels):
		x1, y1, x2, y2 = box
		# Draw bounding box
		cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
		# Add label and score
		text = f"{label}: {score:.2f}"
		cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

	# Display the image with bounding boxes
	plt.imshow(image)
	plt.savefig(image_path.split('.')[0] + '_test.jpg')
	plt.axis('off')
	plt.show()


def visualize_segmentation(image_path, mask):
	# Load image and mask
	image = np.array(Image.open(image_path))

	# Generate random colors for each class
	np.random.seed(0)  # Ensure consistent colors across different runs
	colors = np.random.randint(0, 255, size=(np.max(mask) + 1, 3), dtype=np.uint8)

	# Create a transparent mask overlay
	overlay = np.zeros_like(image, dtype=np.uint8)
	for label in np.unique(mask):
		if label > 0:  # Skip background (label 0)
			overlay[mask == label] = colors[label]

	# Blend the image and the overlay
	alpha = 0.6  # Adjust transparency here (0: fully transparent, 1: fully opaque)
	blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

	# Display the images
	plt.figure(figsize=(12, 6))

	# Original image
	plt.subplot(1, 2, 1)
	plt.imshow(image)
	plt.title('Original Image')
	plt.axis('off')

	# Segmentation overlay
	plt.subplot(1, 2, 2)
	plt.imshow(blended)
	plt.title('Segmentation Overlay')
	plt.axis('off')

	plt.tight_layout()
	plt.show()
