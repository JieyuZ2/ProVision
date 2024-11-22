"""
This code is largely based on https://github.com/jshilong/GPT4RoI/blob/main/gpt4roi/datasets/vcr.py
"""
import copy
import json
import os
import random
from typing import List
from tqdm import tqdm
import re
from tkinter import N

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import path

from osprey.datasets.multi_region import MultiRegionDataset
from osprey.train.train import preprocess, preprocess_multimodal

CHOICES = ['A', 'B', 'C', 'D']

WHY_QUESTIONS = [
    'why?',
    'why',
    "What's the rationale for your decision?",
    'What led you to that conclusion?',
    "What's the reasoning behind your opinion?",
    'Why do you believe that to be true?',
    'Can you explain the basis for your thinking?',
    'What factors influenced your perspective?',
    'How did you arrive at that perspective?',
    'What evidence supports your viewpoint?',
    'What makes you think that way?',
    "What's the logic behind your argument?",
    'Can you provide some context for your opinion?',
    "What's the basis for your assertion?",
    'Why do you hold that belief?',
    'What experiences have shaped your perspective?',
    'What assumptions underlie your reasoning?',
    "What's the foundation of your assertion?",
    "What's the source of your reasoning?",
    "What's the motivation behind your decision?",
    "What's the impetus for your belief?",
    "What's the driving force behind your conclusion?",
    'Why do you think that?',
    "What's your reasoning?",
    'What makes you say that?',
    'Why do you feel that way?',
    "What's the story behind that?",
    "What's your thought process?",
    "What's the deal with that?",
    "What's the logic behind it?",
    'Why do you believe that?',
    "What's the real deal here?",
    "What's the reason behind it?",
    "What's the thought process behind your decision?",
    "What's the rationale for your opinion?",
    'Why do you have that impression?',
    "What's the background to that?",
    "What's the evidence that supports your view?",
    "What's the explanation for that?"
]

MC_ANSWER=[
    "Answer with the option's letter from the given choices directly."
]

Ref_WAY = [
    'There are <region> in the image, ',
    'There are some regions <region>, ',
    'Given <region>, ',
    'Given <region> in the image, ',
    '<region>, ',
    'Several regions <region> are in the image, ',
    '<region> in the given image, '
]

def _spaced_points(low, high,n):
    """ We want n points between low and high, but we don't want them to touch either side"""
    padding = (high-low)/(n*2)
    return np.linspace(low + padding, high-padding, num=n)

def make_mask(height, width, box, polygons_list):
    """
    Mask size: int about how big mask will be
    box: [x1, y1, x2, y2, conf.]
    polygons_list: List of polygons that go inside the box
    """
    mask = np.zeros((height, width), dtype=np.bool_)
    
    xy = np.meshgrid(_spaced_points(box[0], box[2], n=width),
                     _spaced_points(box[1], box[3], n=height)) 
    xy_flat = np.stack(xy, 2).reshape((-1, 2))

    for polygon in polygons_list:
        polygon_path = path.Path(polygon)
        mask |= polygon_path.contains_points(xy_flat).reshape((height, width))
    return mask.astype(np.float32)

def choices_to_text(choices: List[str]) -> str:
    assert len(choices) == len(CHOICES)
    choice_text = ''
    for i,t in zip(CHOICES, choices):
        choice_text += f'{i}. {t}\n'
    return choice_text

class VCRDataset(MultiRegionDataset):
    CLASSES = ('object',)

    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_regions=40,
                 region_mode='segmentation',
                 mc_only=False,
                 use_bbox_text=False,
                 debug=False
                 ):
        
        self.mc_only = mc_only
        self.debug = debug
        super(VCRDataset, self).__init__( 
            tokenizer, data_args, ann_file, img_prefix, 
            max_regions=max_regions, region_mode=region_mode, use_bbox_text=use_bbox_text
        )

        # self.img_prefix = img_prefix
        # self.tokenizer = tokenizer
        # self.data_args = data_args
        # self.mc_only = mc_only

        # self.begin_str = """<image>.\nThis provides an overview of the picture.\n"""
        # self.data_infos = self.load_annotations(ann_file)

    def load_annotations(self, ann_file):

        data_infos = []

        with open(ann_file, 'r') as f:
            for line in tqdm(f):
                ann = json.loads(line)
                metadata_fn_path = ann['metadata_fn']
                img_fn = ann['img_fn']
                img_path = os.path.join(self.img_prefix,img_fn)
                metadata_fn_path = os.path.join(self.img_prefix, metadata_fn_path)
                class_names = ann['objects']

                # Open-Ended Question Answering
                if not self.mc_only:
                    qa_s = []

                    q = ann['question']
                    q = self.replace_numbers_with_tags_tokens(q, class_names)
                    a = ann['answer_choices'][ann['answer_label']]
                    a = self.replace_numbers_with_tags_tokens(a, class_names)
                    why = ann['rationale_choices'][ann['rationale_label']]
                    why = self.replace_numbers_with_tags_tokens(why, class_names)

                    qa_s.append({'from': 'human', 'value': q})
                    qa_s.append({'from': 'gpt', 'value': a})
                    qa_s.append({'from': 'human', 'value': random.choice(WHY_QUESTIONS)})
                    qa_s.append({'from': 'gpt', 'value': why})

                    data_infos.append(dict(
                        img_path = img_path,
                        metadata_path=metadata_fn_path,
                        labels= class_names,
                        qas = qa_s)
                    )

                # MC Format
                qa_s = []
                q = ann['question']
                q = self.replace_numbers_with_tags_tokens(q, class_names)
                answer_choices = [self.replace_numbers_with_tags_tokens(a, class_names) for a in ann['answer_choices']]
                answer_choices = choices_to_text(answer_choices)
                answer = CHOICES[ann['answer_label']]
                rationale_prefix = random.choice(WHY_QUESTIONS)
                rationale_choices = [self.replace_numbers_with_tags_tokens(a, class_names) for a in ann['rationale_choices']]
                rationale_choices = choices_to_text(rationale_choices)
                rationale = CHOICES[ann['rationale_label']]

                qa_s.append({'from': 'human', 'value': q + '\n' + answer_choices + random.choice(MC_ANSWER)})
                qa_s.append({'from': 'gpt', 'value': answer})
                qa_s.append({'from': 'human', 'value': rationale_prefix+ '\n' + rationale_choices + random.choice(MC_ANSWER)})
                qa_s.append({'from': 'gpt', 'value': rationale})

                data_infos.append(dict(
                    img_path = img_path,
                    metadata_path=metadata_fn_path,
                    labels= class_names,
                    qas = qa_s
                ))

                if self.debug:
                    if len(data_infos) > 10:
                        break

        return data_infos
    
    @staticmethod
    def get_regions(bboxes, masks, w, h):
        pred_masks = np.zeros((len(masks), h, w))
        for i,mask in enumerate(masks):

            int_box =  [round(box) for box in bboxes[i]]
            
            height_ = int(int_box[3]-int_box[1])
            width_ = int(int_box[2]-int_box[0])
            box_mask = make_mask(height_, width_, bboxes[i], mask)

            pred_masks[i, int_box[1]:int_box[3], int_box[0]:int_box[2]] = box_mask

        return pred_masks

    @staticmethod 
    def replace_numbers_with_tags(s: str, class_names: List[str]):
        pattern = r'\b(\d+)\b'
        try:
            result = re.sub(pattern, lambda match: f'{class_names[int(match.group(1))-1]} at region{match.group(1)}', s)
        except Exception:
            # contain number not for instance
            return None
        return result

    @staticmethod
    def replace_numbers_with_tags_tokens(tokens: list, class_names: List[str]) -> str:

        result_tokens = []
        for token in tokens:
            if isinstance(token ,list):
                for id in token:
                    region_token = f'{class_names[id]} at region{id+1}'
                    result_tokens.append(region_token)
            else:
                result_tokens.append(token)
        
        result = ' '.join(result_tokens)

        # remove space punctuations
        result = re.sub(r'\s(?=[,.?!])', '', result)

        return result

    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, i, debug=False):
        data_info = self.data_infos[i]

        img_path = data_info['img_path']
        metadata_path = data_info['metadata_path']
        with open(metadata_path, 'r') as metadata_path:
            metadata = json.load(metadata_path)
            masks = metadata['segms']
            bboxes = np.array(metadata['boxes']) # [x1, y1, x2, y2, score]
            bboxes = bboxes[:, :4] # [x1, y1, x2, y2]

        qas = data_info['qas']

        image, image_size, image_token_len = self.process_image(img_path)
        w, h = image_size
        pred_masks = self.get_regions(bboxes, masks, w, h)
        pred_masks = pred_masks[:self.max_regions]
        bboxes = bboxes[:self.max_regions]

        qas = copy.deepcopy(qas)

        # Add image and region prefix
        num_objects = len(pred_masks)
        if self.use_bbox_text:
            if bboxes is None:
                bbox_texts = [self.mask_to_bbox_text(mask) for mask in pred_masks]
            else:
                bbox_texts = [self.bbox_to_text(bbox, h, w) for bbox in bboxes]
            region_string = self.get_region_string(num_objects, bbox_texts)
        else:
            region_string = self.get_region_string(num_objects)
        qas[0]['value'] = self.begin_str + region_string + qas[0]['value']

        sources = preprocess_multimodal(
            copy.deepcopy([qas]),
            self.data_args, image_token_len
        )
        
        if debug:
            for conv in sources[0]:
                print(conv['from'])
                print(conv['value'])
                print("=")

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict['input_ids'][0],
                             labels=data_dict['labels'][0])

        data_dict['image'] = image
        data_dict['masks'] = torch.Tensor(pred_masks)
    
        return data_dict

if __name__ == '__main__':
    from transformers import AutoTokenizer, CLIPImageProcessor
    from osprey.model.multimodal_encoder.clip_encoder import get_image_processor
    from types import SimpleNamespace
    import cv2
    import supervision as sv

    def draw_image(image: torch.Tensor):
        image = image.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = Image.fromarray((image * 255).astype(np.uint8))
        image.save('vcr_image.jpg')
    
    def draw_segmentation(idx: int):
        info =  dataset.data_infos[idx]
        data = dataset.__getitem__(idx, debug=True)
        img_path = info['img_path']
        im = cv2.imread(img_path)

        metadata_path = info['metadata_path']
        with open(metadata_path, 'r') as metadata_path:
            metadata = json.load(metadata_path)
            bboxes = np.array(metadata['boxes'])[:,:4] # [x1, y1, x2, y2, score]
        
        mask = np.array(data['masks'].numpy(), dtype=bool)
        ids = np.array(range(1,len(mask)+1))
        labels = [f"[{idx}]" for idx in ids]
        detections = sv.Detections(xyxy=bboxes, mask=mask, class_id=ids)
        box_annotator = sv.BoxAnnotator()
        annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

        annotated_image = im.copy()
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        cv2.imwrite('vcr_region.jpg',annotated_image)
        
    tokenizer = AutoTokenizer.from_pretrained('../models/Osprey-7b/')
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')
    dataset = VCRDataset(tokenizer, data_args=data_args, ann_file="data/vcr/train.jsonl", img_prefix="../images/vcr/vcr1images", mc_only=True, debug=False, use_bbox_text=True)
    data = dataset[0]
    draw_segmentation(0)
    breakpoint()
