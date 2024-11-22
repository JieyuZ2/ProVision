import copy
import json
from typing import List, Dict, Literal
import os
import random
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import path
from matplotlib import pyplot as plt

from .sg import SGDataset
from osprey.train.train import preprocess, preprocess_multimodal

SG_QUESTIONS = [
    'Generate a scene graph with relevant objects to answer the question using a single word or phrase.',
    'Create a scene graph highlighting objects critical for responding to the question using a single word or phrase.',
    'Develop a scene graph emphasizing objects essential for answering the question correctly.',
    'Build a scene graph featuring objects relevant to formulating an answer to the question.'
]

QUESTIONS = [
    'Answer the question using a single word or phrase.',
]

Ref_WAY = [
    'There are <region> in the image,',
    'There are some regions <region>,',
    'Given <region>,',
    'Given <region> in the image,',
    '<region>,',
    'Several regions <region> are in the image,',
    '<region> in the given image,'
]

def get_xyxy(obj):
    return [obj['x'], obj['y'], obj['x']+obj['w'], obj['y']+obj['h']]

def bboxToMask(box, height, width):
    """
    Mask size: int about how big mask will be
    box: [x1, y1, x2, y2]

    Returns:
    - mask: A numpy array of shape (height, width) with the box area filled with True
    """
    mask = np.zeros((height, width), dtype=np.bool_)

    # Ensure the coordinates are within the image bounds
    x1, y1, x2, y2 = box
    x2, y2 = [min(x2, width), min(y2, height)]
    
    # Fill the box area with True values
    mask[y1:y2, x1:x2] = True
    
    return mask.astype(np.float32)

class TextCoTSGDataset:
    '''
    Perform Scene Graph CoT via bounding box coordinates in text output. 
    Supports if image does not come with segmentation mask.
    '''

    BEGIN_STR = "<image>\nThis provides an overview of the picture and <region1> <mask><pos> highlighting the entire image.\n"

    def get_mask(self, w, h):
        return np.ones((1, h, w))

class GQACoTSGDataset(SGDataset):
    CLASSES = ('object',)

    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=99,
                 max_attributes_per_obj=5,
                 max_relations_per_obj=5,
                 max_qa=10,
                 use_relevant_relations=False,
                 is_train=True,
                 region_mode: Literal['box', 'segmentation']='segmentation',
                 ignored_relations: List[str] = None,
                 sg_mode: int=2,
                 no_cot=False,
                 use_bbox_text=False,
                 ):

        self.max_qa = max_qa
        self.region_mode = region_mode        
        self.no_cot = no_cot
        self.use_relevant_relations = use_relevant_relations
        super().__init__(tokenizer, data_args, ann_file, img_prefix, 
                         max_gt_per_img=max_gt_per_img, max_attributes_per_obj=max_attributes_per_obj, max_relations_per_obj=max_relations_per_obj, 
                         is_train=is_train, region_mode=region_mode, ignored_relations=ignored_relations, sg_mode=sg_mode, use_bbox_text=use_bbox_text)

    def load_annotations(self, ann_file):

        with open(ann_file, 'r') as f:
          ann_dict = json.load(f)
        data_infos = []

        for image_id, ann in tqdm(ann_dict.items()):
            width = ann['width']
            height = ann['height']
            sg: dict = ann['scene_graph']
            object_mapping = ann['object_mapping']

            # assign ids to scene graph objects
            sg_keys = list(sg.keys())
            sg_keys = sg_keys[:self.max_gt_per_img]
            if self.is_train:
                random.shuffle(sg_keys)
            object2id = {k: (idx+1) for idx, k in enumerate(sg_keys)} # {'object_id': one-indexed id}
            region_mapping = {k: object2id[v] for k,v in object_mapping.items() if v in object2id} # {'gqa_object_id': one-indexed id}
            
            segmentations = []
            boxes = []
            for obj in sg_keys:
                if obj not in object2id:
                    continue

                info: dict = sg[obj]

                # Add segmentation regions
                segmentations.append(info['segmentation'])
                boxes.append(info['bbox'])
            
            assert len(segmentations) > 0
            assert len(boxes) == len(segmentations), \
                "number of boxes: ({}) and segmentations: ({}) should match".format(len(boxes), len(segmentations))

            img_path = os.path.join(self.img_prefix,image_id+'.jpg')

            no_cot = self.no_cot
            if no_cot:
                q = random.choice(QUESTIONS)
            else:
                q = random.choice(SG_QUESTIONS)
            begin_string = q
            sg_s = []

            # SG CoT per QA
            for idx, qa in enumerate(ann['data'][:self.max_qa]):
                objects = qa['object_ids'] # GQA object ids

                output_text = ''
                if not no_cot:
                    object_ids = sorted([object_mapping[obj] for obj in objects if obj in object_mapping], 
                                        key=lambda x: object2id[x]) # GQA object to index ids
                    # Actually just shuffle it lol
                    if False:
                        random.shuffle(object_ids)
                    
                    if self.use_relevant_relations:
                        sg_mapping = {k: v for k,v in region_mapping.items() if k in objects}
                    else:
                        sg_mapping = region_mapping
                    
                    for obj in object_ids: 
                        region_id: int = object2id[obj]
                        info = sg[obj]
                        sg_dict = self.get_sg_dict(info, sg_mapping)
                        sg_text = f"region{region_id}: {sg_dict}"
                        # output_text += f"{sg_texts[obj]}\n"
                        output_text += f"{sg_text}\n"
                output_text += f"Answer: {qa['answer']}"

                prefix = begin_string + ' ' if idx == 0 else ''
                sg_s.append({'from': 'human', 'value': prefix + qa['question']})
                sg_s.append({'from': 'gpt', 'value': output_text})

            data_infos.append(dict(
                img_path = img_path,
                boxes = boxes, 
                segmentations=segmentations,
                convs = sg_s)
            )

        return data_infos

class GQACoTGroundedSGDataset(GQACoTSGDataset):
    CLASSES = ('object',)

    def load_annotations(self, ann_file):

        with open(ann_file, 'r') as f:
          ann_dict = json.load(f)
        data_infos = []

        for image_id, ann in tqdm(ann_dict.items()):
            sg: dict = ann['scene_graph']

            regions = ann['regions']
            regions = regions[:self.max_gt_per_img]

            id_region_mapping = ann['id_region_mapping']
            object_mapping = {k: (idx+1) for k, idx in id_region_mapping.items() if idx < len(regions)} # {object_id to region_id} [+ 1 for one-indexed id]

            boxes, segmentations = self.process_regions(regions)

            img_path = os.path.join(self.img_prefix,image_id+'.jpg')

            no_cot = self.no_cot or ann['id_region_mapping'] is None
            if no_cot:
                q = random.choice(QUESTIONS)
            else:
                q = random.choice(SG_QUESTIONS)
            begin_string = q
            sg_s = []

            # SG CoT per QA
            for idx, qa in enumerate(ann['data'][:self.max_qa]):
                object_ids = qa['object_ids'] # object_ids in scene graph for specific QA

                output_text = ''
                if not no_cot:
                    region_sg = {}
                    for object_id in object_ids: 
                        if object_id in object_mapping: # if grounded object
                            region_id: int = object_mapping[object_id]
                            info = sg[object_id]
                            sg_dict: dict = self.get_sg_dict(info, object_mapping)
                            region_sg[region_id] = sg_dict
                    output_text += self.textify_region_sg(region_sg)
                
                output_text += f"Answer: {qa['answer']}"

                prefix = begin_string + ' ' if idx == 0 else ''
                sg_s.append({'from': 'human', 'value': prefix + qa['question']})
                sg_s.append({'from': 'gpt', 'value': output_text})

            data_infos.append(dict(
                img_path = img_path,
                boxes = boxes, 
                segmentations=segmentations,
                convs = sg_s)
            )

        return data_infos

if __name__ == '__main__':
    from transformers import AutoTokenizer, CLIPImageProcessor
    from osprey.model.multimodal_encoder.clip_encoder import get_image_processor
    from types import SimpleNamespace
    import cv2
    import supervision as sv
    tokenizer = AutoTokenizer.from_pretrained('../models/Osprey-7b/')
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')
    dataset = GQACoTGroundedSGDataset(
        tokenizer, data_args=data_args, 
        ann_file="data/gqa/train_cot_gqa_sam_seem_regions_grounded.json", 
        img_prefix="../images/gqa/images",
        max_relations_per_obj=5,
        region_mode='box_segmentation',
        sg_mode=2,
        no_cot=False,
        ignored_relations=['to the left of', 'to the right of'],
        use_relevant_relations=True,
        use_bbox_text=True,
        )
    data = dataset.__getitem__(0, debug=True)

    def draw_segmentation(idx: int):
        info =  dataset.data_infos[idx]
        img_path = info['img_path']
        im = cv2.imread(img_path)
        boxes = np.array(info['boxes'])

        data = dataset.__getitem__(idx, debug=True)
        mask = np.array(data['masks'].numpy(), dtype=bool)
        ids = np.array(range(len(mask)))
        labels = [f"[{idx+1}]" for idx in ids]
        detections = sv.Detections(xyxy=boxes, mask=mask, class_id=ids)
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

        annotator = sv.MaskAnnotator()
        annotated_image = im.copy()
        # annotated_image = box_annotator.annotate(scene=im.copy(), detections=detections, labels=labels)
        annotated_image = annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        cv2.imwrite('gqa_cot_sg.jpg',annotated_image)
    
    draw_segmentation(0)
    breakpoint()
    print(dataset.data_infos[1]['sgs'][1]['value'])
    data = []
    bad_id = [] 
    for idx in tqdm(range(len(dataset))):
        try:
            data.append(dataset.__getitem__(idx))
        except Exception:
            bad_id.append(idx)
    print('bad ids: {}'.format(bad_id))
    breakpoint()
    import pandas as pd
    pd.DataFrame([len(d['input_ids']) for d in data]).describe()
