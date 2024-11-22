import copy
import random
import os
import cv2
import json
from tqdm import tqdm
from typing import List, Dict, Literal
from pathlib import Path
from panopticapi.utils import rgb2id
import numpy as np
import torch

from .sg import SGDataset
from .prompts import SG_QUESTIONS

def get_region_id(id: str, region_mapping: Dict[str, int]) -> int:
    """Utility function to map region IDs."""
    return int(region_mapping[id]) + 1  # (1-indexed)

class PSGDataset(SGDataset):
    CLASSES = ('object',)

    def __init__(
            self,
            tokenizer,
            data_args=None,
            ann_file=None,
            img_prefix=None,
            seg_img_prefix=None,
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

        self.seg_img_prefix = seg_img_prefix
        super().__init__(tokenizer, data_args, ann_file, img_prefix,
                         max_gt_per_img=max_gt_per_img, max_relations_per_obj=max_relations_per_obj,
                         is_train=is_train, region_mode=region_mode, ignored_relations=ignored_relations,
                         sg_mode=sg_mode, shuffle_relations=shuffle_relations,
                         use_bbox_text=use_bbox_text
                         )
    
    @staticmethod
    def load_image(image) -> np.ndarray:
        img_bgr = cv2.imread(str(image))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    @classmethod
    def create_segmentations(cls, pan_seg_image, segments_info, ) -> np.ndarray:
        im = rgb2id(cls.load_image(pan_seg_image))
        segmentations = []
        for segment_info in segments_info:
            mask = im == segment_info['id']
            segmentations.append(mask)
            assert np.sum(mask) > 0
            
        return np.array(segmentations, dtype=bool)

    def get_psg_object(self, object_class: str) -> str:
        query = object_class.replace('-merged','')
        query = query.replace('-other','')
        query = query.replace('-stuff','')
        query = query.replace('-',' is ')

        return query.strip()


    def load_annotations(self, ann_file):

        with open(ann_file, 'r') as f:
            ann = json.load(f)

        predicate_classes = ann['predicate_classes']
        object_classes = ann['thing_classes'] + ann['stuff_classes']
        
        data_infos = []
        data = ann['data']
        for idx, datum in enumerate(tqdm(data)):
            image_id = str(datum['image_id'])
            height = datum['height']
            width = datum['width']
            img_path = os.path.join(self.img_prefix, datum['file_name'])

            # Gather region segmentations
            objects = datum['annotations']
            segments_info = datum['segments_info']

            assert len(objects) == len(segments_info)

            objects = objects[:self.max_regions]
            segments_info = segments_info[:self.max_regions]

            boxes = np.array([a['bbox'] for a in objects]) # xyxy
            pan_seg_image = os.path.join(self.seg_img_prefix, datum['pan_seg_file_name'])
            segs: np.ndarray = self.create_segmentations(pan_seg_image, segments_info)

            if len(boxes) == 0:
                continue
            
            assert len(boxes) == len(segs), \
                "number of boxes: ({}) and segmentations: ({}) should match".format(len(boxes), len(segs))

            # Build Scene Graph with region_idx (1-indexed): info
            sg = {idx+1: obj for idx,obj in enumerate(objects)} # + 1 for 1-index
            for k,v in sg.items():
                object_class = object_classes[v['category_id']]
                name = self.get_psg_object(object_class)
                v['name'] = name
                v['relations'] = []

            # Add relations
            relations: list = datum['relations']
            for relation in relations:
                subj, obj, rel_class = relation
                subj = subj + 1
                obj = obj + 1
                sg[subj]['relations'].append({'object': obj, 'name': predicate_classes[rel_class]})

            # Create convrsation
            sg_s = self.generate_conversation(sg)
            if sg_s:
                data_infos.append(dict(
                    img_path = img_path,
                    boxes = boxes, 
                    segmentations=segs,
                    convs = sg_s)
                )
            
        return data_infos
     
    def generate_conversation(self, region_sg: Dict[int, Dict]):
        # Create convrsation
        sg_s = []

        q = random.choice(SG_QUESTIONS)
        final_sg_text: str = self.textify_region_sg(region_sg)
        sg_s.append({'from': 'human', 'value': q})
        sg_s.append({'from': 'gpt', 'value': final_sg_text})
    
        return sg_s

    def textify_region_sg(self, region_sg: Dict[int, Dict]) -> str:

        def textify_relation(relation: dict) -> str:
            return f"region{relation['object']} {relation['name']}"

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
            relations = sg_dict.get('relations', [])[:self.max_relations_per_obj]
            relations = sorted(relations, key=lambda x: x['object'])
            if len(relations) > 0:
                text_relations = [textify_relation(relation) for relation in relations]
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

            final_obj_text =  f"Objects:\n{final_obj_text}"
            final_rel_text = f"Relations:\n{final_relation_text}"

            return f"{final_obj_text}\n{final_rel_text}"
        
        return None
        

    def __len__(self):
        return len(self.data_infos)

if __name__ == '__main__':
    from transformers import AutoTokenizer, CLIPImageProcessor
    from osprey.model.multimodal_encoder.clip_encoder import get_image_processor
    from types import SimpleNamespace
    import supervision as sv
    def draw_segmentation(idx: int):
        info =  dataset.data_infos[idx]
        data = dataset.__getitem__(idx, debug=True)
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
        cv2.imwrite('psg.jpg',annotated_image)

    tokenizer = AutoTokenizer.from_pretrained('../models/Osprey-7b/')
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')
    dataset = PSGDataset(tokenizer, data_args=data_args, 
                        ann_file="../data/psg/psg_train_val.json", 
                        img_prefix="../images/coco/", 
                        seg_img_prefix="../data/coco/panoptic_annotations")
    data_info = dataset.get_data_info(0)
    draw_segmentation(0)
    breakpoint()


