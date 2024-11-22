import copy
import json
from typing import List, Dict, Literal
import os
import random
from tqdm import tqdm

import numpy as np

from .multi_region import MultiRegionDataset
from .prompts import SG_QUESTIONS, RELATION_QUESTIONS

def get_xyxy(obj):
    return [obj['x'], obj['y'], obj['x']+obj['w'], obj['y']+obj['h']]

class SGDataset(MultiRegionDataset):
    CLASSES = ('object',)

    def __init__(
            self,
            tokenizer,
            data_args=None,
            ann_file=None,
            img_prefix=None,
            max_gt_per_img=99,
            max_attributes_per_obj=5,
            max_relations_per_obj=5,
            
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

            img_path = os.path.join(self.img_prefix,image_id+'.jpg')

            # Create conversation
            sg_s = self.generate_conversation(region_sg)
            if sg_s:
                data_infos.append(dict(
                    img_path = img_path,
                    boxes = boxes, 
                    segmentations=segmentations,
                    convs = sg_s)
                )

        return data_infos

    def generate_conversation(self, region_sg: Dict[int, Dict]):
        """
        Generates single round conversation data for scene graph generation.

        Args:
            region_sg (Dict[int, Dict]): A dictionary representing the region-specific information.

        Returns:
            List[Dict[str, str]]: A list of dictionaries representing the conversation between the human and the AI agent.
        """
        sg_s = []

        q = random.choice(SG_QUESTIONS)
        final_sg_text: str = self.textify_region_sg(region_sg)
        sg_s.append({'from': 'human', 'value': q})
        sg_s.append({'from': 'gpt', 'value': final_sg_text})

        return sg_s
    
    def textify_relation(self, relation: dict) -> str:
        return f"region{relation['object']} {relation['name']}"

    def textify_region_sg(self, region_sg: Dict[int, Dict]) -> str:

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
                obj_text = f"{obj_text} is {att_text}"
            obj_texts[region_id] = obj_text

            # Gather relations
            relations = sg_dict.get('relations', [])
            relations = relations[:self.max_relations_per_obj]
            relations = sorted(relations, key=lambda x: x['object'])
            if len(relations) > 0:
                text_relations = [self.textify_relation(relation) for relation in relations]
                text_relations = ', '.join(text_relations)
                relation_texts[region_id] = text_relations
        
        if self.sg_mode == 1: # Not recommended probably
            final_text = ""
            for region_id in region_ids:
                region_text = f"region{region_id}: {obj_texts[region_id]}"
                if region_id in relation_texts:
                    region_text += f"\nRelations: {relation_texts[region_id]}"
                final_text += f"{region_text}\n"
            return final_text

        elif self.sg_mode == 2:

            final_obj_text = ""
            final_relation_text = ""
            for region_id in region_ids:
                final_obj_text += f"region{region_id}: {obj_texts[region_id]}\n"
                if region_id in relation_texts:
                    final_relation_text += f"region{region_id}: {relation_texts[region_id]}\n"

            final_text = f"Objects:\n{final_obj_text}"
            if final_relation_text:
                final_rel_text = f"Relations:\n{final_relation_text}"
                final_text = f"{final_text}\n{final_rel_text}"

            return final_text
        
        elif self.sg_mode == 3: # dictionary format

            final_text = ""
            for region_id in region_ids:
                region_dict = region_sg[region_id]
                region_dict_str = ', '.join(f"{k}: {v}" for k, v in region_dict.items())
                final_text += f"region{region_id}: {region_dict_str}\n"
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
        relations = sorted(relations, key=lambda x: x['object'])
        relations = relations[:self.max_relations_per_obj]

        # if sort_relations: # Sort relations by object ID that appear in
        #     relations = sorted(relations, key=lambda x: x['object'])

        sg_dict= {'name': name, 'attributes': attributes, 'relations': relations}

        # remove key-values that are empty to free up tokenization space
        for k in list(sg_dict.keys()):
            if isinstance(sg_dict[k], list) and len(sg_dict[k]) == 0:
                sg_dict.pop(k)
        
        return sg_dict

    def __len__(self):
        return len(self.data_infos)
class RelationDataset(SGDataset):

    def generate_conversation(self, region_sg: Dict[int, Dict], begin_string: str):

        # Create conversation
        sg_s = []

        subject_ids = list(region_sg.keys())
        if self.is_train:
            random.shuffle(subject_ids)

        for subj_id in subject_ids:

            sg_dict = region_sg[subj_id]
            subject_region = f'region{subj_id}'

            # List of Relations as Text
            relations = sg_dict.get('relations', [])
            relations = sorted(relations, key=lambda x: x['object'])
            if len(relations) > 0:
                text_relations = [self.textify_relation(relation) for relation in relations]
                relation_text = ', '.join(text_relations)
            else:
                continue

            if len(sg_s) == 0:
                q = random.choice(RELATION_QUESTIONS).format(subject_region)
                sg_s.append({'from': 'human', 'value': begin_string + ' ' + q})
            else:
                sg_s.append({'from': 'human', 'value': subject_region})
            sg_s.append({'from': 'gpt', 'value': relation_text})
    
        return sg_s

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
        mask = np.array(data['masks'].numpy(), dtype=bool)
        ids = np.array(range(len(mask)))
        labels = [f"[{idx+1}]" for idx in ids]
        detections = sv.Detections(xyxy=boxes, mask=mask, class_id=ids)
        box_annotator = sv.BoxAnnotator()
        annotator = sv.MaskAnnotator()
        annotated_image = box_annotator.annotate(scene=im.copy(), detections=detections, labels=labels)
        annotated_image = annotator.annotate(scene=annotated_image, detections=detections)
        cv2.imwrite('vg_sg.jpg',annotated_image)

    tokenizer = AutoTokenizer.from_pretrained('../models/Osprey-7b/')
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')
    for i in [2]:

        # dataset = RelationDataset(
        #     tokenizer, data_args=data_args, 
        #     ann_file='data/sg/vg_train_sg_sam_hq_v3.json', # 'data/sg/vg_train_sg_sam_hq.json',
        #     img_prefix="../images/gqa/images",
        #     ignored_relations=['to the left of', 'to the right of'],
        #     is_train=True,
        #     max_gt_per_img=100,
        #     max_relations_per_obj=5,
        #     sg_mode=i
        # )

        dataset = SGDataset(
            tokenizer, data_args=data_args, 
            # ann_file='data/sg/vg_train_sg_sam_hq.json',
            ann_file='data/sg/psg_vg_train_sg_sam_hq_v4_full.json',
            img_prefix="../images/gqa/images",
            ignored_relations=['to the left of', 'to the right of'],
            is_train=True,
            max_gt_per_img=99,
            max_relations_per_obj=5,
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
