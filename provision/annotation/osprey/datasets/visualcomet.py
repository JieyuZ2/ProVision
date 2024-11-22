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
from osprey.train.train import preprocess, preprocess_multimodal
from osprey.datasets.vcr import VCRDataset

EVENT_QUESTIONS = [
    "What can you tell me about {} in the image?",
    "What is/are {} doing in the image?",
    "Provide details about {} in the image.",
    "Summarize what {} is/are doing in the image.",
    "Detail the activities of {} in the image.",
    "Describe the event taking place in the image involving {}.",
    "Explain the event that {} is part of in the image.",
    "Provide details about the event in the image featuring {}.",
    "What event is {} involved in within the image?",
]

INTENT_QUESTIONS = [
    "What is the intention of the person at {} in the image?",
    "What is the person at {} trying to achieve in the image?",
    "What is the person at {} aiming to do in the image?",
    "What is the objective of the person at {} in the image?",
    "What is the person at {} hoping to accomplish in the image?",
    "What is the purpose of the person at {} in the image?",
]

BEFORE_QUESTIONS = [
    "What would have the person at {} needed to do before in the image?",
    "What actions did the person at {} need to take before this moment in the image?",
    "What steps did the person at {} need to complete before this event in the image?",
    "What prior activities did the person at {} engage in before this scene in the image?",
    "What did the person at {} do leading up to this moment in the image?",
    "What events occurred before this moment for the person at {} in the image?",
    "What preparations did the person at {} make before this scene in the image?",
    "What led to the current situation for the person at {} in the image?",
]

AFTER_QUESTIONS = [
    "What would the person at {} most likely do after in the image?",
    "What will the person at {} probably do after this moment in the image?",
    "What subsequent actions will the person at {} take in the image?",
    "What is the next step for the person at {} in the image?",
    "What will the person at {} do following this moment in the image?",
    "What is likely to happen next for the person at {} in the image?",
    "What future actions will the person at {} take in the image?",
    "What will the person at {} do after this event in the image?",
]

LIST_ANSWER = [
    "Answer as a list of actions.",
    "Provide the answer in a list format.",
    "List the actions as the answer.",
    "Respond with a list of steps.",
    "Give the answer as a sequence of actions.",
    "Detail the answer in a list of activities.",
    "Enumerate the actions in the answer.",
    "Present the answer as a series of actions.",
]

class VisualCometDataset(VCRDataset):
    CLASSES = ('object',)

    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_regions=30,
                 use_event=True,
                 use_inference=True,
                 combine_inference=True,
                 shuffle_inference=True,
                 person_only=True,
                 use_bbox_text=False,
                 debug=False
                 ):

        self.use_event = use_event
        self.use_inference = use_inference
        self.person_only = person_only
        self.combine_inference = combine_inference  # Combine all inferences into one question
        self.shuffle_inference = shuffle_inference # Shuffle the order of inferences
        self.bad_ids = []
        super(VCRDataset, self).__init__(
            tokenizer, data_args, ann_file, img_prefix, 
            max_regions=max_regions, use_bbox_text=use_bbox_text
        )
    
    def subject_to_text(self, subject: list[int], class_names: list[str]) -> str:
        ''' subject: 1-indexed list of class ids'''
        if not subject:
            return 'people'
        return ', '.join(map(lambda x: f'{class_names[x-1]} at region{x}' if isinstance(x, int) else x, subject)) # 0 to 1 index mapping
    
    @staticmethod
    def replace_numbers_with_tags_tokens(tokens: list, class_names: List[str]) -> str:
        """ Unlike VCR that uses 0-indexing, visualcomet uses 1-indexing for regions"""

        result_tokens = []
        for token in tokens:
            if isinstance(token ,list):
                for id in token:
                    assert id > 0, f"Region ID should be greater than 0, got {id}"
                    region_token = f'{class_names[id-1]} at region{id}'
                    result_tokens.append(region_token)
            else:
                result_tokens.append(token)
        
        result = ' '.join(result_tokens)

        # remove space punctuations
        result = re.sub(r'\s(?=[,.?!])', '', result)

        return result
    
    def load_annotations(self, ann_file):

        data_infos = []

        with open(ann_file, 'r') as f:
            for idx, line in enumerate(tqdm(f)):
                ann = json.loads(line)
                metadata_fn_path = ann['metadata_fn']
                img_fn = ann['img_fn']
                img_path = os.path.join(self.img_prefix,img_fn)
                metadata_fn_path = os.path.join(self.img_prefix, metadata_fn_path)

                # load metdata info
                class_names = ann['objects']
                if self.person_only: # Only keep the person in the image
                    class_names = [class_name for class_name in class_names if class_name == 'person']
    
                subjects: str = self.subject_to_text(ann['subject'], class_names)

                # Events
                if self.use_event:
                    qa_s = []
                    try:
                        a = self.replace_numbers_with_tags_tokens(ann['event'], class_names)
                    except IndexError:
                        print(ann['event'], len(class_names))
                        print(f"Event Error in {ann['image_id']}")
                        self.bad_ids.append(ann['image_id'])
                        continue

                    q = random.choice(EVENT_QUESTIONS).format(subjects)
                    qa_s.append({'from': 'human', 'value': q})
                    qa_s.append({'from': 'gpt', 'value': a})
                    data_infos.append(dict(
                        img_path = img_path,
                        metadata_path=metadata_fn_path,
                        labels= class_names,
                        qas = qa_s)
                    )

                # Inferences
                if self.use_inference:
                    inference_types = ['intent', 'before', 'after']
                    if self.shuffle_inference:
                        random.shuffle(inference_types)
                    
                    if self.combine_inference:
                        qa_s = []
                        for idx,inference_type in enumerate(inference_types):
                            q = random.choice(eval(f'{inference_type.upper()}_QUESTIONS')).format(subjects)
                            inferences: list[list] = ann[inference_type]
                            
                            try:
                                inferences = [self.replace_numbers_with_tags_tokens(inference, class_names) for inference in inferences]
                            except IndexError:
                                print(inferences, len(class_names))
                                print(f"Inference Error in {ann['image_id']}")
                                self.bad_ids.append(ann['image_id'])
                                continue
                            a = "; ".join(inferences)

                            if idx == 0: # Add list prompt only in the beginning
                                q = q + ' ' + random.choice(LIST_ANSWER)

                            qa_s.append({'from': 'human', 'value': q})
                            qa_s.append({'from': 'gpt', 'value': a})

                        data_infos.append(dict(
                            img_path = img_path,
                            metadata_path=metadata_fn_path,
                            labels= class_names,
                            qas = qa_s)
                        )
                    else:
                        raise NotImplementedError("Separate Inference Loading Not implemented yet")

        return data_infos
    
    def __getitem__(self, i, debug=False):
        data_info = self.data_infos[i]

        img_path = data_info['img_path']
        metadata_path = data_info['metadata_path']
        labels = data_info['labels']
        num_objects = len(labels)
        with open(metadata_path, 'r') as metadata_path:
            metadata = json.load(metadata_path)
            masks = metadata['segms'][:num_objects]
            bboxes = np.array(metadata['boxes'])[:num_objects] # [x1, y1, x2, y2, score]
            bboxes = bboxes[:, :4]


        image, image_size, image_token_len = self.process_image(img_path)
        w, h = image_size
        pred_masks = self.get_regions(bboxes, masks, w, h)
        pred_masks = pred_masks[:self.max_regions]
        bboxes = bboxes[:self.max_regions]

        qas = data_info['qas']
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
    tokenizer = AutoTokenizer.from_pretrained('/net/nfs.cirrascale/mosaic/jamesp/models/Osprey-7b/')
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')
    dataset = VisualCometDataset(tokenizer, data_args=data_args, 
                                 ann_file="data/visualcomet/train_annots_gpt-4o_parsed_clean.jsonl", 
                                 img_prefix="/net/nfs.cirrascale/mosaic/jamesp/images/vcr/vcr1images/",
                                 use_bbox_text=True
            )
    data = dataset.__getitem__(0)
    breakpoint()
