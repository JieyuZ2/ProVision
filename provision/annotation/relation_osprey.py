import json
import os
import pickle as pkl
from functools import partial

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm, trange
from transformers import AutoTokenizer, CLIPImageProcessor

from .osprey.constants import IMAGE_TOKEN_INDEX
from .osprey.conversation import SeparatorStyle, conv_templates
from .osprey.mm_utils import tokenizer_image_token
from .osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM
from .osprey.train.train import DataArguments

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

styles_with_single_sep = [SeparatorStyle.SINGLE, SeparatorStyle.MPT, SeparatorStyle.PLAIN]

REF_WAY_NUM = [
	'There are {} regions in the image: <region>.\n',
	'There are {} part regions in the image, given <region>.\n',
	'The image contains {} regions, including <region>.\n',
	'In the image, there are {} regions, such as <region>.\n',
	'This image displays {} regions, including <region>.\n',
	'Among the {} regions in the image, there is <region>.\n',
	'The picture has {} regions, one of which is <region>.\n',
	'You can see {} regions in the image, like <region>.\n',
	'There are {} distinct regions in the image, such as <region>.\n',
	'The image features {} regions, including <region>.\n',
]


class Osprey:
	def __init__(self,
				 model_path: str,
				 clip_path: str,
				 device: str = 'cuda',
				 detection_mode: str = "holistic"):

		setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
		setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

		model_path = os.path.expanduser(model_path)
		self.tokenizer = AutoTokenizer.from_pretrained(
			model_path,
			model_max_length=2048,
			padding_side="right",
			use_fast=True
		)
		self.tokenizer.pad_token = self.tokenizer.unk_token
		self.device = device
		if device == 'cpu':
			self.model = OspreyLlamaForCausalLM.from_pretrained(
				model_path,
				mm_vision_tower=clip_path).to(device)

			vision_tower = self.model.get_vision_tower()
			if not vision_tower.is_loaded:
				vision_tower.load_model()
			vision_tower.to(device=device)
		else:
			self.model = OspreyLlamaForCausalLM.from_pretrained(
				model_path,
				mm_vision_tower=clip_path,
				torch_dtype=torch.bfloat16).to(device)

			vision_tower = self.model.get_vision_tower()
			if not vision_tower.is_loaded:
				vision_tower.load_model()
			vision_tower.to(dtype=torch.float16, device=device)

		self.image_processor = CLIPImageProcessor(do_resize=True,
												  size={"shortest_edge": 512},
												  resample=3,
												  do_center_crop=True,
												  crop_size={"height": 512, "width": 512},
												  do_rescale=True, rescale_factor=0.00392156862745098,
												  do_normalize=True,
												  image_mean=[0.48145466, 0.4578275, 0.40821073],
												  image_std=[0.26862954, 0.26130258, 0.27577711],
												  do_convert_rgb=True)

		spi_tokens = ['<mask>', '<pos>']
		self.tokenizer.add_tokens(spi_tokens, special_tokens=True)
		self.detection_mode = detection_mode

		for m in self.model.modules():
			m.tokenizer = self.tokenizer

	def process_image(self, image_list: Image):
		image_aspect_ratio = self.model.config.image_aspect_ratio
		if image_aspect_ratio == "pad":

			def expand2square(pil_img, background_color):
				width, height = pil_img.size
				if width == height:
					return pil_img
				elif width > height:
					result = Image.new(pil_img.mode, (width, width), background_color)
					result.paste(pil_img, (0, (width - height) // 2))
					return result
				else:
					result = Image.new(pil_img.mode, (height, height), background_color)
					result.paste(pil_img, ((height - width) // 2, 0))
					return result

			image_list = [expand2square(img, tuple(int(x * 255) for x in self.image_processor.image_mean)) for img in image_list]
			image_list = self.image_processor.preprocess(image_list, return_tensors="pt", do_center_crop=False)["pixel_values"]  # (batch_size, 3, 512, 512)
		else:
			image_list = self.image_processor.preprocess(image_list, return_tensors="pt", do_center_crop=False)["pixel_values"]
			image_list = torch.nn.functional.interpolate(image_list,
														 size=(512, 512),
														 mode='bilinear',
														 align_corners=False)

		return image_list

	def get_region_string(self, n: int, bbox_labels: list[str] = None):
		"""
		Generates a string of region references based on the number of regions and optional bbox texts.

		Args:
			n (int): The number of regions.
			bbox_labels (list[str], optional): A list of bbox texts, same size as `n`. Defaults to None.

		Returns:
			str: A string of region references.
		"""
		if bbox_labels is not None:
			assert len(bbox_labels) == n, f"Length of bbox_labels ({len(bbox_labels)}) must be equal to n ({n})."

		ref_string = ''
		for i in range(n):
			if not bbox_labels:
				ref_string = ref_string + f'region{i + 1} <mask><pos>, '
			else:
				ref_string = ref_string + f'region{i + 1} <mask><pos> {bbox_labels[i]}, '
		ref_string = ref_string[:-2]  # remove the last comma

		# ref_prefix = random.choice(REF_WAY)
		ref_prefix = REF_WAY_NUM[0].format(n)
		region_string = ref_prefix.replace('<region>', ref_string)

		return region_string

	def init_prompt(self,
					num_of_seg_per_instance=2,
					obj_idx=None,
					bbox_labels=[]):
		rel_prompt = self.get_region_string(num_of_seg_per_instance, bbox_labels)
		begin_str = """<image>\nThis provides an overview of the picture.\n"""

		if obj_idx is None:
			# qs = begin_str + rel_prompt + init_qs_prompt + ' and '.join([f'region{obj_idx + 1}' for i in range(num_of_seg_per_instance)]) + '?'
			qs = begin_str + rel_prompt + ' Generate scene graph for given regions.'
		else:
			qs = begin_str + rel_prompt + f' What is the relationship between all regions and region{obj_idx + 1}?'

		conv = conv_templates['osprey_v1'].copy()
		conv.append_message(conv.roles[0], qs)
		conv.append_message(conv.roles[1], None)
		self.stop_str = conv.sep if conv.sep_style in styles_with_single_sep else conv.sep2

		prompt = conv.get_prompt()
		input_ids = tokenizer_image_token(prompt,
										  self.tokenizer,
										  IMAGE_TOKEN_INDEX,
										  return_tensors='pt').unsqueeze(0).to(self.model.device)
		return input_ids

	def generate_relation(self, input_ids, masks, image):
		with torch.inference_mode():
			self.model.orig_forward = self.model.forward
			if self.model.device.type != 'cpu':
				self.model.forward = partial(self.model.orig_forward,
											 img_metas=[None],
											 masks=[masks.half()])

				output_ids = self.model.generate(input_ids,
												 images=image.half(),
												 do_sample=True,
												 temperature=0.2,
												 top_p=0.95,
												 max_new_tokens=1024,
												 use_cache=True,
												 num_beams=1)
			else:
				self.model.forward = partial(self.model.orig_forward,
											 img_metas=[None],
											 masks=[masks])

				output_ids = self.model.generate(input_ids,
												 images=image,
												 do_sample=True,
												 temperature=0.2,
												 top_p=0.95,
												 max_new_tokens=1024,
												 use_cache=True,
												 num_beams=1)
			self.model.forward = self.model.orig_forward

		input_token_len = input_ids.shape[1]
		n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()

		if n_diff_input_output > 0:
			print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

		outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:],
											  skip_special_tokens=True)[0]

		outputs = outputs.strip()
		if outputs.endswith(self.stop_str):
			outputs = outputs[:-len(self.stop_str)]

		return outputs

	def relation_detection(self,
						   image_list: list,
						   seg_annotation_path: str,
						   seg_dir_path: str,
						   cache_file: str = None,
						   ):  # holistic or detailed
		res = []
		seg_annotation = json.load(open(seg_annotation_path, 'r'))

		if os.path.exists(cache_file):
			cache = json.load(open(cache_file, 'r'))
		else:
			cache = {}

		pointer = 0
		bs = 100
		with trange(len(image_list), desc="Relation Detection: ", unit="steps", ncols=150, position=0, leave=True) as pbar:

			while True:
				pil_image_list, img_ids = [], []
				for image_path in image_list[pointer:pointer + bs]:
					img_id = image_path.split('/')[-1].split('.')[0]
					if img_id not in cache:
						img_ids.append(img_id)
						pil_image_list.append(Image.open(image_path).convert('RGB'))

				if len(pil_image_list) == 0:
					continue
				processed_image_list = self.process_image(pil_image_list)  # (batch_size, 3, 512, 512)

				for img, img_id in zip(processed_image_list, img_ids):
					annotation = seg_annotation[img_id]['annotation']  # [n_bboxes, w, h]
					bbox_labels = annotation['labels']
					mask_path = os.path.join(seg_dir_path, annotation['seg_mask_id'])
					if ".pkl" in mask_path:
						compressed_mask = pkl.load(open(os.path.join(seg_dir_path, annotation['seg_mask_id']), 'rb'))
						raw_masks = np.array([c_masks.toarray() for c_masks in compressed_mask], dtype=bool)
					else:
						raw_masks = np.load(os.path.join(seg_dir_path, annotation['seg_mask_id']))

					img = img.unsqueeze(0).to(self.device)
					masks = torch.from_numpy(np.array(raw_masks)).to(self.device)
					img_relations = []

					try:
						if self.detection_mode == "holistic":
							input_ids = self.init_prompt(num_of_seg_per_instance=len(raw_masks), bbox_labels=bbox_labels)
							outputs = self.generate_relation(input_ids, masks, img)
							img_relations = outputs

						elif self.detection_mode == "detailed":
							for idx in range(len(raw_masks)):
								input_ids = self.init_prompt(num_of_seg_per_instance=len(raw_masks), obj_idx=idx, bbox_labels=bbox_labels)
								outputs = self.generate_relation(input_ids, masks, img)
								img_relations.append(outputs)

						else:
							raise f"Invalid detection mode {self.detection_mode}"

					except Exception as e:
						print(f"Error in image {img_id}")
						raise e


					res.append(img_relations)
					cache[img_id] = img_relations
					if (len(cache) + 1) % 100 == 0:
						print(f"Save cache to {cache_file}: {len(cache)}")
						json.dump(cache, open(cache_file, 'w'), indent=2)

					pbar.update()

				pointer += bs
				if pointer >= len(image_list):
					break

		return [{
			"raw_relations": r,
		} for r in res]
