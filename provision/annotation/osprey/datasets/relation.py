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

from .cot_sg import SGDataset
from osprey.train.train import preprocess, preprocess_multimodal

RELATION_QUESTIONS = [
    'Generate list of relationships for: {}.',
    'Assign relations for: {}',
    'Can you assign relations to objects in {}?',
    'How can we map all relationships for {}?',
    'What are the all connections you see in {}?',
    'Identify the inter-regional relationships for {}.',
    'Could you detail the interactions for {}?',
    'Please outline the network of relationships for {}.',
    'Can you delineate the ties binding {}?',
    'Could you classify the types of relationships present in {}?',
]

RELATION_DESCRIPTION_QUESTIONS = [
    'Generate a description for: {}.',
    'Describe {} in details.',
    'What is goign on in {}?',
    'Give a summary of {}.',
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


class RelationDataset(SGDataset):
    CLASSES = ('object',)

    def __init__(
            self,
            tokenizer,
            data_args=None,
            ann_file=None,
            img_prefix=None,
            max_gt_per_img=150,
            max_relations_per_obj=10,
            
            is_train=True,
            region_mode: Literal['box', 'segmentation', 'box_segmentation']='segmentation',
            ignored_relations: List[str] = None,
            sg_mode: int=1,
            shuffle_relations: bool = False,
            use_bbox_text=False,
    ):

        self.is_train = is_train
        self.begin_str = """<image>.\nThis provides an overview of the picture.\n"""
        self.ignored_relations = [] if ignored_relations is None else ignored_relations
        self.use_box = region_mode == 'box'
        self.max_relations_per_obj = max_relations_per_obj
        self.sg_mode = sg_mode
        self.shuffle_relations = shuffle_relations
        
        super().__init__(tokenizer, data_args, ann_file, img_prefix, 
                         max_gt_per_img=max_gt_per_img, max_relations_per_obj=max_relations_per_obj,
                         use_bbox_text=use_bbox_text
                         )


        print('{} (mode {}): {}'.format(self.__class__.__name__, self.sg_mode, len(self.data_infos)))
    
    """
        Annotation Format:
        {
            'image_id': image_id,
            'width': width,
            'height': height,
            'regions': [{}],
            '
        }
    """

    def load_annotations(self, ann_file):

        with open(ann_file, 'r') as f:
            ann_list = []
            for line in f:  
                ann_list.append(json.loads(line))
        data_infos = []

        for ann in tqdm(ann_list):

            image_id = ann['image_id']
            id_region_mapping: Dict[str, int] = ann['id_region_mapping'] # Object ID to Relation Index

            # Try keeping regions mentioned in the relations.
            mentioned_regions = list(set(id_region_mapping.values()))
            if self.is_train and random.random() < 0.5:
                region_map = {v: idx for idx, v in enumerate(mentioned_regions)}
            else:
                region_map = {idx: idx for idx in range(len(ann['regions']))}
            
            # Process candidate regions
            boxes = []
            segmentations = []
            for idx,region in enumerate(ann['regions']):

                if idx not in region_map:
                    continue
            
                # add region segmentations
                segmentations.append(region['segmentation'])
                boxes.append(region['xyxy'])
            
            if len(boxes) == 0:
                continue
            
            assert len(boxes) == len(segmentations), \
                "number of boxes: ({}) and segmentations: ({}) should match".format(len(boxes), len(segmentations))

            img_path = os.path.join(self.img_prefix, image_id+'.jpg')

            # Add region prompts (1-indexed)
            num_objects = len(segmentations)
            ref_string = ''
            for i in range(num_objects):
                ref_string = ref_string +  f'region{i+1} <mask><pos>' + ','
            ref_string = ref_string[:-1]
            ref_prefix = random.choice(Ref_WAY)

            begin_string = ref_prefix.replace('<region>', ref_string)

            # Create relation conversation
            sg_s = []
            subj: str = ann['subject']
            relations: List[List[str, str]] = ann['relations'][:self.max_relations_per_obj]

            # Prompt asking about all relationships for subject
            def get_region_id(id):
                return region_map[int(id_region_mapping[id])] + 1  # (1-indexed)

            region_subj = get_region_id(subj)
            assert (region_subj-1) in region_map.values()

            # Create relationship list for subject
            rel_q = random.choice(RELATION_QUESTIONS).format(f'region{region_subj}')
            rel_text = []
            for relation in relations:
                rel_name, obj_id = relation
                region_obj = get_region_id(obj_id)
                assert (region_obj-1) in region_map.values()
                rel_text.append(f"{rel_name} region{region_obj}")
            rel_text = ', '.join(rel_text)
            sg_s.append({'from': 'human', 'value': begin_string + ' ' + rel_q})
            sg_s.append({'from': 'gpt', 'value': rel_text})

            data_infos.append(dict(
                img_path = img_path,
                boxes = boxes, 
                segmentations=segmentations,
                sgs = sg_s)
            )
        
        return data_infos

class RelationDescriptionDataset(RelationDataset):

    def load_annotations(self, ann_file):

        with open(ann_file, 'r') as f:
            ann_list = []
            for line in f:  
                ann_list.append(json.loads(line))
        data_infos = []

        for ann in tqdm(ann_list):

            image_id = ann['image_id']
            id_region_mapping: Dict[str, int] = ann['id_region_mapping'] # Object ID to Relation Index

            # Try keeping regions mentioned in the relations.
            mentioned_regions = list(set(id_region_mapping.values()))
            if self.is_train and random.random() < 0.5:
                region_map = {v: idx for idx, v in enumerate(mentioned_regions)}
            else:
                region_map = {idx: idx for idx in range(len(ann['regions']))}
            
            # Process candidate regions
            boxes = []
            segmentations = []
            for idx,region in enumerate(ann['regions']):

                if idx not in region_map:
                    continue
            
                # add region segmentations
                segmentations.append(region['segmentation'])
                boxes.append(region['xyxy'])
            
            if len(boxes) == 0:
                continue
            
            assert len(boxes) == len(segmentations), \
                "number of boxes: ({}) and segmentations: ({}) should match".format(len(boxes), len(segmentations))

            img_path = os.path.join(self.img_prefix, image_id+'.jpg')

            # Add region prompts (1-indexed)
            num_objects = len(segmentations)
            ref_string = ''
            for i in range(num_objects):
                ref_string = ref_string +  f'region{i+1} <mask><pos>' + ','
            ref_string = ref_string[:-1]
            ref_prefix = random.choice(Ref_WAY)

            begin_string = ref_prefix.replace('<region>', ref_string)

            # Create relation conversation
            sg_s = []
            subj: str = ann['subject']

            # Prompt asking about all relationships for subject
            def get_region_id(id):
                return region_map[int(id_region_mapping[id])] + 1  # (1-indexed)

            region_subj = get_region_id(subj)
            assert (region_subj-1) in region_map.values()

            rel_q = random.choice(RELATION_DESCRIPTION_QUESTIONS).format(f'region{region_subj}')
            rel_text = ann['description']
            sg_s.append({'from': 'human', 'value': begin_string + ' ' + rel_q})
            sg_s.append({'from': 'gpt', 'value': rel_text})

            data_infos.append(dict(
                img_path = img_path,
                boxes = boxes, 
                segmentations=segmentations,
                sgs = sg_s)
            )
        
        return data_infos

if __name__ == '__main__':
    from transformers import AutoTokenizer, CLIPImageProcessor
    from types import SimpleNamespace
    from osprey.model.multimodal_encoder.clip_encoder import get_image_processor
    import cv2
    import supervision as sv

    def draw_segmentation(idx: int):
        info =  dataset.data_infos[idx]
        data = dataset.__getitem__(idx)
        sg = info['sgs']
        img_path = info['img_path']
        im = cv2.imread(img_path)
        boxes = np.array(info['boxes'])
        mask = np.array(data['masks'].numpy(), dtype=bool)
        ids = np.array(range(len(mask)))
        labels = [f"[{idx+1}]" for idx in ids]
        detections = sv.Detections(xyxy=boxes, mask=mask, class_id=ids)[:10]
        box_annotator = sv.BoxAnnotator()
        annotator = sv.MaskAnnotator()
        annotated_image = box_annotator.annotate(scene=im.copy(), detections=detections, labels=labels)
        annotated_image = annotator.annotate(scene=annotated_image, detections=detections)
        cv2.imwrite('vg_sg.jpg',annotated_image)

    tokenizer = AutoTokenizer.from_pretrained('models/Osprey-7b/')
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')
    dataset = RelationDataset(
        tokenizer, data_args=data_args, 
        ann_file='data/relation/train_coco_relation_category_sam_seem_regions_150.jsonl',
        img_prefix="/mmfs1/gscratch/raivn/jspark96/data/images/gqa/images",
        is_train=True,
        # region_mode='box',
    )
    draw_segmentation(0)
    breakpoint()
    print(dataset.data_infos[1]['sgs'][1]['value'])
    bad_id = [] 
    for idx in tqdm(range(len(dataset))):
        try:
            dataset.__getitem__(idx)
        except Exception:
            bad_id.append(idx)
    print('bad ids: {}'.format(bad_id))
    breakpoint()
