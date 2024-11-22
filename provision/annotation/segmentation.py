import os
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .utils import get_image, prepare_sam_model, save_mask_as_png, save_sep_mask

try:
	from sam2.build_sam import build_sam2
	from sam2.sam2_image_predictor import SAM2ImagePredictor
	from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
	import warnings

	warnings.warn("segment_anything is not installed. Annotation functions can not be used.")


def sam_mask_prediction(
		image_list: list[str],
		path_to_checkpoint: str = "sam2_hiera_large.pt",
		save_path: str = "./seg_pred_res/",
		device: str = "cuda",
		sam2_version: str = "072824",
		sam2_config_path: str = "sam2_hiera_l.yaml",
		input_points: Optional[List[np.ndarray]] = None,
		input_labels: Optional[List[np.ndarray]] = None,
		bboxes: Optional[List[np.ndarray]] = None,
		labels: Optional[List[str]] = None,
		multimask_output: Optional[bool] = False,
		save_png: bool = False,
		overwrite: bool = False,
		**kwargs
):
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	prepare_sam_model(path_to_checkpoint, version=sam2_version)
	predictor = SAM2ImagePredictor(build_sam2(sam2_config_path, path_to_checkpoint))

	with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
		res = []
		for image_path_or_url, image_bboxes, image_label in tqdm(zip(image_list, bboxes, labels), desc="segmentation prediction", total=len(image_list)):
			if "http" in image_path_or_url:
				image_name = image_path_or_url.split('/')[-1][:os.path.basename(image_path_or_url).rfind(".")]
			else:
				image_name = os.path.basename(image_path_or_url)[:os.path.basename(image_path_or_url).rfind(".")]
			whole_mask_path = f"{save_path}/{image_name}_mask.pkl"

			if not os.path.exists(whole_mask_path) or overwrite:
			# if os.path.exists(whole_mask_path) and overwrite:
			# 	print(f"Overwriting {whole_mask_path}")

				raw_image = np.array(get_image(image_path_or_url))
				predictor.set_image(raw_image)

				# If multimask_output been set to True, it will return three masks with three scores that represent their quality.
				# Output format: Tuple[masks, scores, logits]
				masks = []
				for bbox in image_bboxes:
					mask = predictor.predict(
						point_coords=input_points,
						point_labels=input_labels,
						box=np.array(bbox),
						multimask_output=multimask_output,
						**kwargs
					)[0]
					masks.append({
						"segmentation": mask[0],
						"bbox"        : bbox
					})

				whole_mask_path, _ = save_sep_mask(masks, whole_mask_path)

				if save_png:
					save_mask_as_png(masks, f"{save_path}/{image_name}_mask.png")

			res.append({
				"seg_mask_id": whole_mask_path.split('/')[-1],
				"bboxes"     : image_bboxes,
				"labels"     : image_label
			})
	return res

import pickle as pkl

def sam_mask_generation(
		image_list: list[str],
		path_to_checkpoint: str = "sam2_hiera_large.pt",
		save_path: str = "./seg_pred_res",
		device: str = "cuda",
		sam2_version: str = "072824",
		sam2_config_path: str = "sam2_hiera_l.yaml",
		save_png: bool = False
):
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	prepare_sam_model(path_to_checkpoint, version=sam2_version)
	mask_generator = SAM2AutomaticMaskGenerator(build_sam2(sam2_config_path, path_to_checkpoint))

	with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
		res = []
		for image_path_or_url in tqdm(image_list, desc="segmentation generation", total=len(image_list)):
			if "http" in image_path_or_url:
				image_name = image_path_or_url.split('/')[-1][:os.path.basename(image_path_or_url).rfind(".")]
			else:
				image_name = os.path.basename(image_path_or_url)[:os.path.basename(image_path_or_url).rfind(".")]
			whole_mask_path = f"{save_path}/{image_name}_mask.pkl"

			if not os.path.exists(whole_mask_path):

				raw_image = np.array(get_image(image_path_or_url))
				masks = mask_generator.generate(raw_image)

				whole_mask_path, _ = save_sep_mask(masks, whole_mask_path)

				if save_png:
					save_mask_as_png(masks, f"{save_path}/{image_name}_mask_all.png")

			res.append({
				"seg_mask_id": whole_mask_path.split('/')[-1],
			})

		return res
