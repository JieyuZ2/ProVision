import json
import copy
import os
import random
import re
from typing import List,Dict, Literal
from tqdm import tqdm
from PIL import Image
import torch 

from osprey.train.train import preprocess, preprocess_multimodal
from .llava import LlavaDataset, xywh2xyxy
from .prompts import GROUNDED_QA
from osprey.constants import PHRASE_START, PHRASE_END

class GromaDataset(LlavaDataset):
    
    def parse_text(self, s: str):
        s = s.strip()
        s = s.replace('> ', '>')
        s = s.replace(' </', '</')
        s = s.replace('><', '> <')
        return s
        
    
    def load_annotations(self, ann_file):

        st = "<p>"
        ed = "</p>"
        box_token = "<ground_box>"

        ann_list = []
        with open(ann_file, 'r') as f:
            ann_list = json.load(f)

        data_infos = []
        for idx,ann in enumerate(tqdm(ann_list)):

            image_id = ann['file_name']
            img_path = os.path.join(self.img_prefix, image_id)

            img_w = ann['width']
            img_h = ann['height']

            boxes = [xywh2xyxy(box) for box in ann['boxes']]

            qa_s = []
            for i in range(len(ann['conversation'])//2):
                    
                question: str = ann['conversation'][i*2]['value']
                if i == 0:
                    question = random.choice(GROUNDED_QA) + ' ' + question
                answer: str = ann['conversation'][i*2+1]['value']

                # Strip unnecessary tokens
                answer = answer.replace(st, PHRASE_START)
                answer = answer.replace(ed, PHRASE_END)
                answer = answer.replace('<roi>', '')
                answer = answer.replace('</roi>', '')

                boxes_seq = ann['conversation'][i*2+1]['box_inds']  
                assert len(boxes_seq) == answer.count(box_token)
                for box_seq in boxes_seq:
                    boxes_list = np.array([boxes[box_seq]])
                    bbox_text = self.bbox_to_text(boxes_list, img_h, img_w)
                    answer = answer.replace(box_token, bbox_text, 1)
                
                assert answer.count(st) == 0 and answer.count(ed) == 0

                answer = self.parse_text(answer)
                qa_s.append({'from': 'human', 'value': question})         
                qa_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = img_path,
                boxes = boxes,
                convs = qa_s
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
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        cv2.imwrite('refexp.jpg',annotated_image)

    dataset = GromaDataset(
        tokenizer, 
        data_args=data_args, 
        ann_file="../data/groma/groma_instruct.json", 
        img_prefix="../images/vg/VG_100K_all/",
    )

    data_info = dataset.get_data_info(0)
    draw_segmentation(0)
    breakpoint()


