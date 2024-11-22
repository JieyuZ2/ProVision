import json
import copy
import os
import random
from typing import List,Dict, Literal
from tqdm import tqdm
from PIL import Image
import torch 

from osprey.train.train import preprocess, preprocess_multimodal
from .llava import LlavaDataset
from .prompts import GROUNDED_DESCRIPTION_QUESTIONS
from osprey.constants import PHRASE_START, PHRASE_END

class FlickrEntitiesDataset(LlavaDataset):
    
    def load_annotations(self, ann_file):

        st = "<ph_st>"
        ed = "<ph_ed>"

        ann_list = []
        with open(ann_file, 'r') as f:
            for line in f:  
                ann_list.append(json.loads(line))

        data_infos = []
        for idx,ann in enumerate(tqdm(ann_list)):

            image_id = str(ann['image_id'])
            img_path = os.path.join(self.img_prefix, image_id+'.jpg')

            img_w = ann['img_w']
            img_h = ann['img_h']

            question = random.choice(GROUNDED_DESCRIPTION_QUESTIONS)
            sentence: str = ann['sentence']
            boxes = ann['boxes']
            boxes_seq: list[list[int]] = ann["boxes_seq"] 

            assert len(boxes_seq) == sentence.count(ed)
            for idx, box_seq in enumerate(boxes_seq):
                sentence = sentence.replace(st, PHRASE_START, 1)
                boxes_list = [boxes[s] for s in box_seq]
                bbox_text = self.bbox_to_text(boxes_list, img_h, img_w)
                sentence = sentence.replace(ed, PHRASE_END + ' ' + bbox_text, 1)
            
            assert sentence.count(st) == 0 and sentence.count(ed) == 0

            sg_s = []
            sg_s.append({'from': 'human', 'value': question})
            sg_s.append({'from': 'gpt', 'value': sentence})

            data_infos.append(dict(
                img_path = img_path,
                boxes = boxes,
                convs = sg_s
            ))
            
        
        return data_infos

if __name__ == '__main__':
    from transformers import AutoTokenizer, CLIPImageProcessor
    from osprey.model.multimodal_encoder.clip_encoder import get_image_processor
    from types import SimpleNamespace
    import numpy as np
    import cv2
    import supervision as sv
    tokenizer = AutoTokenizer.from_pretrained('../models/Osprey-7b/')
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')

    def draw_segmentation(idx: int):
        info =  dataset.data_infos[idx]
        img_path = info['img_path']
        data = dataset.__getitem__(idx, debug=True)

        im = cv2.imread(img_path)
        boxes = np.array(info['boxes'])
        ids = np.array(range(len(boxes)))
        labels = [f"[{idx+1}]" for idx in ids]
        detections = sv.Detections(xyxy=boxes, class_id=ids)
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

        annotated_image = im.copy()
        annotated_image = box_annotator.annotate(scene=im.copy(), detections=detections)
        # annotated_image = annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        cv2.imwrite('flickr.jpg',annotated_image)

    dataset = FlickrEntitiesDataset(
        tokenizer, 
        data_args=data_args, 
        ann_file="../data/flickr/flickr30k_train.jsonl", 
        img_prefix="../data/flickr/flickr30k_images/flickr30k_images/",
    )

    data_info = dataset.get_data_info(0)
    draw_segmentation(0)
    breakpoint()


