from collections import defaultdict
import copy
import json
import re
from typing import List, Dict, Literal
import os
import random
from tqdm import tqdm

import torch
from PIL import Image
import numpy as np

from .multi_region import MultiRegionDataset
from .prompts import SG_QUESTIONS, REGION_PROPOSAL_QUESTIONS, SG_DETECTION_QUESTIONS

from osprey.train.train import preprocess, preprocess_multimodal

def get_xyxy(obj):
    return [obj['x'], obj['y'], obj['x']+obj['w'], obj['y']+obj['h']]

class SGDetectionDataset(MultiRegionDataset):
    CLASSES = ('object',)

    def __init__(
            self,
            tokenizer,
            data_args=None,
            ann_file=None,
            img_prefix=None,
            max_gt_per_img=99,
            max_attributes_per_obj=5,
            max_relations_per_obj=10,
            
            is_train=True,
            region_mode: Literal['box', 'segmentation', 'box_segmentation']='segmentation',
            ignored_relations: List[str] = None,
            sg_mode: int=2,
            shuffle_relations: bool = False,
            use_bbox_text=False,
    ):

        self.ignored_relations = [] if ignored_relations is None else ignored_relations
        self.max_attributes_per_obj = max_attributes_per_obj
        self.max_relations_per_obj = max_relations_per_obj
        self.sg_mode = sg_mode
        self.shuffle_relations = shuffle_relations
        
        super().__init__(tokenizer, data_args, ann_file, img_prefix,
                         max_regions=max_gt_per_img, max_gt_per_img=max_gt_per_img,
                         is_train=is_train, region_mode=region_mode,
                         use_bbox_text=use_bbox_text
                         )


        print('{} (mode {}): {}'.format(self.__class__.__name__, self.sg_mode, len(self.data_infos)))
    
    """
        Annotation Format:
        {
            'image_id': {
                'width': width,
                'height': height,
                'objects': {
                    'obj_id': {
                        'name': name,
                        'attributes': attributes,
                        'relations': [{'object': other_object, 'name': relation_name}],
                        'bbox': [xyxy],
                        'segmentation': sam_segmentation
                    }
                }
            }
        }
    """

    def load_annotations(self, ann_file):

        with open(ann_file, 'r') as f:
          ann_dict = json.load(f)
        data_infos = []

        for image_id, ann in tqdm(ann_dict.items()):
            sg: dict = ann['objects']

            # assign ids to scene graph objects
            sg_keys = list(sg.keys())
            if self.is_train:
                random.shuffle(sg_keys)
            sg_keys = sg_keys[:self.max_gt_per_img]
            object2id = {k: (idx+1) for idx, k in enumerate(sg_keys)} # {'object_id': one-indexed region id}
            
            segmentations = []
            boxes = []
            region_sg: Dict[int, Dict] = {}
            for obj in sg_keys:
                if obj not in object2id:
                    continue
                info = sg[obj]

                # create sg for region
                sg_dict: dict = self.get_sg_dict(info, object2id)

                region_id: int = object2id[obj]
                region_sg[region_id] = sg_dict

                # add region segmentations
                segmentations.append(info['segmentation'])
                boxes.append(info['bbox'])
            
            if len(boxes) == 0:
                continue
            
            assert len(boxes) == len(segmentations), \
                "number of boxes: ({}) and segmentations: ({}) should match".format(len(boxes), len(segmentations))
            
            assert len(boxes) == len(region_sg)

            img_path = os.path.join(self.img_prefix,image_id+'.jpg')

            # Create conversation
            height = ann['height']
            width = ann['width']
            sg_s = self.generate_conversation(region_sg, boxes, height, width)
            if sg_s:
                data_infos.append(dict(
                    img_path = img_path,
                    boxes = boxes, 
                    segmentations=segmentations,
                    convs = sg_s)
                )

        return data_infos
    
    def generate_conversation_deprecated(self, region_sg: Dict[int, Dict], boxes, height: int, width: int):
        """
        Depcrecated: Generate regions first then create scene graph from regions.
        """
        sg_s = []

        # Region proposal
        q = random.choice(REGION_PROPOSAL_QUESTIONS)
        sg_s.append({'from': 'human', 'value': q})
        bbox_texts = [self.bbox_to_text(bbox, height, width) for bbox in boxes]
        num_objects = len(region_sg)
        region_proposal = self.get_region_string(num_objects, bbox_texts)
        region_proposal = region_proposal.replace('<mask><pos> ', '')
        region_proposal = re.sub(r'\s<mask><pos>\s', ' ', region_proposal).strip() # remove <mask><pos> tokens when outputting region proposal
        sg_s.append({'from': 'gpt', 'value': region_proposal})

        # Scene graph generation
        q = random.choice(SG_QUESTIONS)
        region_string = self.get_region_string(num_objects, None)
        final_sg_text: str = self.textify_region_sg(region_sg)
        sg_s.append({'from': 'human', 'value': region_string + q})
        sg_s.append({'from': 'gpt', 'value': final_sg_text})

        return sg_s
    
    def generate_conversation(self, region_sg: Dict[int, Dict], boxes, height: int, width: int):
        """
        Generates single round conversation data for scene graph generation.

        Args:
            region_sg (Dict[int, Dict]): A dictionary representing the region-specific information.

        Returns:
            List[Dict[str, str]]: A list of dictionaries representing the conversation between the human and the AI agent.
        """
        sg_s = []

        # Region proposal
        q = random.choice(SG_DETECTION_QUESTIONS)
        sg_s.append({'from': 'human', 'value': q})
        bbox_texts = [self.bbox_to_text(bbox, height, width) for bbox in boxes]
        final_sg_text: str = self.textify_region_sg(region_sg, bbox_texts=bbox_texts)
        sg_s.append({'from': 'gpt', 'value': final_sg_text})

        return sg_s
    
    def textify_relation(self, relation: dict) -> str:
        return f"region{relation['object']} {relation['name']}"

    def textify_region_sg(self, region_sg: Dict[int, Dict], bbox_texts=None) -> str:

        obj_texts = {}
        relation_texts = {}

        region_ids = sorted(list(region_sg.keys()))

        # Gather Object Labels
        for region_id in region_ids:

            sg_dict = region_sg[region_id]

            obj_text = sg_dict['name']
            attributes = sg_dict.get('attributes', [])
            if len(attributes) > 0:
                att_text = ', '.join(attributes)
                obj_text = f"{obj_text} in {att_text}"
            obj_texts[region_id] = obj_text

            # Gather relations
            relations = sg_dict.get('relations', [])
            relations = relations[:self.max_relations_per_obj]
            relations = sorted(relations, key=lambda x: x['object'])
            if len(relations) > 0:
                text_relations = [self.textify_relation(relation) for relation in relations]
                text_relations = ', '.join(text_relations)
                relation_texts[region_id] = text_relations
        
        if self.sg_mode == 0: # relations that group by predicates with list of objects 
            final_obj_text = ""
            final_relation_text = ""
            for idx,region_id in enumerate(region_ids):
                if bbox_texts is not None:
                    final_obj_text += f"region{region_id}: {obj_texts[region_id]} {bbox_texts[idx]}\n"
                else:
                    final_obj_text += f"region{region_id}: {obj_texts[region_id]}\n"
                if region_id in relation_texts:
                    relations = defaultdict(list)
                    for relation in region_sg[region_id]['relations']:
                        object_name = f"region{relation['object']}"
                        relations[relation['name']].append(object_name)
                    relation_text = ''
                    for predicate, objects in relations.items():
                        object_text = ', '.join(objects)
                        relation_text += f"{predicate}: {object_text}; "
                    relation_text = relation_text[:-1]
                    final_relation_text += f"region{region_id}: {relation_text}\n"

            final_text = f"Objects:\n{final_obj_text}"
            if final_relation_text:
                final_rel_text = f"Relations:\n{final_relation_text}"
                final_text = f"{final_text}\n{final_rel_text}"

            return final_text

        elif self.sg_mode == 1: # Not recommended probably
            final_text = ""
            for idx, region_id in enumerate(region_ids):
                region_text = f"region{region_id}: {obj_texts[region_id]}"
                if region_id in relation_texts:
                    region_text += f"\nRelations: {relation_texts[region_id]}"
                final_text += f"{region_text}\n"
            return final_text

        elif self.sg_mode == 2:

            final_obj_text = ""
            final_relation_text = ""
            for idx, region_id in enumerate(region_ids):
                if bbox_texts is not None:
                    final_obj_text += f"region{region_id}: {obj_texts[region_id]} {bbox_texts[idx]}\n"
                else:
                    final_obj_text += f"region{region_id}: {obj_texts[region_id]}\n"
                if region_id in relation_texts:
                    final_relation_text += f"region{region_id}: {relation_texts[region_id]}\n"

            final_text = f"Objects:\n{final_obj_text}"
            if final_relation_text:
                final_rel_text = f"Relations:\n{final_relation_text}"
                final_text = f"{final_text}\n{final_rel_text}"

            return final_text
        
        return None
    
    def get_sg_dict(self, info: dict, object_mapping: dict, sort_relations=True) -> dict:
        """
        Generates a scene graph dict from the provided info and object mapping.

        Args:
            info (dict): Contains name, attributes, and relations.
            object_mapping (dict): Maps objects to their regions.

        Returns:
            dict: Scene graph with 'name', 'attributes', and 'relations'. 
                  Note: Empty keys are removed.
        """
        
        name: str = info['name']
        attributes: list = info.get('attributes', [])
        relations = [rel for rel in info.get('relations', [])
                        if isinstance(rel, dict) and rel.get('object', None) in object_mapping \
                        and rel['name'] not in self.ignored_relations]
        for rel in relations:
            rel['object'] = object_mapping[rel['object']]

        attributes = attributes[:self.max_attributes_per_obj]
        
        # process relations per object
        relations= [dict(t) for t in {tuple(d.items()) for d in relations}] # remove duplicates
        relations = sorted(relations, key=lambda x: x['object']) # sort by object id
        relations = relations[:self.max_relations_per_obj] # limit number of relations per object

        sg_dict= {'name': name, 'attributes': attributes, 'relations': relations}

        # remove key-values that are empty to free up tokenization space
        for k in list(sg_dict.keys()):
            if isinstance(sg_dict[k], list) and len(sg_dict[k]) == 0:
                sg_dict.pop(k)
        
        return sg_dict

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, i, debug=False):
        """
        Retrieves an item from the dataset at a given index.

        Args:
            i (int): The index of the item to retrieve.
            debug (bool, optional): If True, provides additional debug information for the conversation. Defaults to False.

        Returns:
            dict: A dictionary containing the processed image, its associated masks, and conversation tokens.

        The expected format of `data_infos` is a list of dictionaries, where each dictionary contains:
            - 'img_path': The path to the image file.
            - 'boxes': Absolute xyxy bounding boxes for objects in the image.
            - 'segmentations': The segmentation masks for objects in the image.
            - 'convs': A list of dictionaries, each representing a conversation.
        """
        data_info = self.data_infos[i]

        img_path = data_info['img_path']
        boxes = data_info['boxes']
        segmentations = data_info['segmentations']
        convs: List[Dict] = data_info['convs']

        image, image_size, image_token_len = self.process_image(img_path)
        w, h = image_size

        # Load masks as bbox, segm, or bbox + segm
        pred_masks: np.ndarray = self.create_masks(boxes, segmentations, h, w)
        convs = copy.deepcopy(convs)

        # Add image and region prefix
        num_objects = len(pred_masks)
        convs[0]['value'] = self.begin_str + convs[0]['value']

        sources = preprocess_multimodal(
            copy.deepcopy([convs]),
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
        data_dict['masks'] = None # torch.Tensor(pred_masks)

        return data_dict


if __name__ == '__main__':
    from transformers import AutoTokenizer, CLIPImageProcessor
    from types import SimpleNamespace
    from osprey.model.multimodal_encoder.clip_encoder import get_image_processor
    import cv2
    import supervision as sv

    def draw_segmentation(idx: int):
        info =  dataset.data_infos[idx]
        data = dataset.__getitem__(idx)
        img_path = info['img_path']
        im = cv2.imread(img_path)
        boxes = np.array(info['boxes'])
        # mask = np.array(data['masks'].numpy(), dtype=bool)
        ids = np.array(range(len(boxes)))
        labels = [f"[{idx+1}]" for idx in ids]
        detections = sv.Detections(xyxy=boxes, class_id=ids)
        box_annotator = sv.BoundingBoxAnnotator()
        annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

        annotated_image = box_annotator.annotate(scene=im.copy(), detections=detections)
        # annotated_image = annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        cv2.imwrite('vg_sg.jpg',annotated_image)

    tokenizer = AutoTokenizer.from_pretrained('../models/Osprey-7b/')
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')
    for i in [2]:

        dataset = SGDetectionDataset(
            tokenizer, data_args=data_args, 
            ann_file='data/sg/psg_vg_train_sg_sam_hq_v4_subset.json',
            img_prefix="../images/gqa/images",
            ignored_relations=['to the left of', 'to the right of'],
            is_train=True,
            max_gt_per_img=99, # number of regions
            max_relations_per_obj=10,
            region_mode='box',
            sg_mode=i
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
