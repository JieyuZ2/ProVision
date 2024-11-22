import numpy as np
import torch
import json
import os
import copy
import random
from .stage2_data import CustomDataset, xywh2xyxy
from osprey.train.train import preprocess, preprocess_multimodal
import re

DETAILED_QUESTIONS =  [
    'Can you provide me with a detailed description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail?',
    'What details can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image.',
    'Can you give me a detailed account of the region labeled as <region> in the picture?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail?",
    'What is the region outlined by <region> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <region> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <region> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image, please.',
    'Can you give me a detailed account of the region labeled as <region> in the picture, please?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a detailed description?',
    'Please describe the region <region> in the image in detail.',
    'Can you offer a thorough analysis of the region <region> in the image?',
    'Could you elaborate on the region highlighted by <region> in the picture provided?',
    'Please share more information about the zone emphasized with <region> in the photo.',
    'What insights can you give ablout the area denoted by <region> in the image presented?',
    'Can you share a comprehensive rundown of the region denoted by <region> in the presented image?',
    "I'd like to know more about the region highlighted by <region> in the picture provided.",
    'Work through the important details of the area <region> in the image.',
    'Illustrate the area represtented by <region> through a descriptive explanation.',
    'Examine the region <region> closely and share its details.'
]


class ConversationDataset(CustomDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 use_bbox_text=False,
                 ):
        self.begin_str = "<image>\nThis provides an overview of the picture.\n"
        super().__init__(tokenizer, data_args, ann_file, img_prefix, use_bbox_text=use_bbox_text)

    def get_region_string(self, n: int, bbox_texts=None):

        if bbox_texts is not None:
            assert len(bbox_texts) == n

        ref_string = ""
        for i in range(n):
            if not bbox_texts:
                ref_string = ref_string +  f'region{i+1} <mask><pos>' + ','
            else:
                ref_string = ref_string +  f'region{i+1} <mask><pos> {bbox_texts[i]}' + ','
        ref_string = ref_string[:-1]
                    
        if n==1:
            mid_str = "There are 1 part region in the picture: "+ref_string+'. '
        else:
            mid_str = "There are {} part regions in the picture: ".format(str(n))+ref_string+'. '
        
        return mid_str

    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file))

        for ann in ann_list:
            if len(ann['conversations'])//2 ==0:
                continue
            bboxes = []
            masks = []
            qa_s = []
            filename = ann['file_name'].split('_')[-1]
            img_path = os.path.join(self.img_prefix, filename)
            region_num = len(ann['annotation'])
            h, w = ann['height'], ann['width']

            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']
                masks.append(mask)

                bbox = xywh2xyxy(ann['annotation'][i]['bbox'])
                bboxes.append(bbox)

            for i in range(len(ann['conversations'])//2):
                    
                question = ann['conversations'][i*2]['value']
                question = question.replace('<','').replace('>','')  
                qa_s.append({'from': 'human', 'value': question})
                
                answer = ann['conversations'][i*2+1]['value']
                answer = answer.replace('<','').replace('>','')
                qa_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = img_path,
                bboxes=bboxes,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s
            ))
        return data_infos

    def __getitem__(self, i, debug=False):
        data_info = self.data_infos[i]
        img_path = data_info['img_path']
        height = data_info['height']
        width = data_info['width']
        bboxes = data_info['bboxes']
        masks_raw = data_info['masks']
        masks = []
        for mask_r in masks_raw:
            mask = self.annToMask(mask_r, height, width)
            masks.append(mask)
        
        assert len(bboxes) == len(masks)
            
        masks = np.array(masks)
        ori_h, ori_w = masks[0].shape
        qas = data_info['qas']
        qas = copy.deepcopy(qas)

        image, image_size, image_token_len = self.process_image(img_path)

        # Add image and region prefix
        num_objects = len(masks)
        if self.use_bbox_text:
            bbox_texts = [self.bbox_to_text(bbox, ori_h, ori_w) for bbox in bboxes]
            region_string = self.get_region_string(num_objects, bbox_texts)
        else:
            region_string = self.get_region_string(num_objects)
        qas[0]['value'] = self.begin_str + region_string + qas[0]['value']

        sources = preprocess_multimodal(
            copy.deepcopy([qas]),
            self.data_args, image_token_len)
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

        data_dict['masks'] = torch.Tensor(masks)

        return data_dict

class OspreyPartLevel(ConversationDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 use_bbox_text=False,
                 ):
        self.limit = ' Answer the question using a single word or phrase.'
        super().__init__(tokenizer, data_args, ann_file, img_prefix, use_bbox_text=use_bbox_text)

class OspreyLVISPosNeg(ConversationDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 use_bbox_text=False,
                 ):
        
        super().__init__(tokenizer, data_args, ann_file, img_prefix, use_bbox_text=use_bbox_text)

    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file))

        for ann in ann_list:
            if len(ann['conversations'])//2 ==0:
                continue

            bboxes = []
            masks = []
            qa_s = []
            filename = ann['file_name']
            img_path = os.path.join(self.img_prefix, filename)
            region_num = len(ann['annotation'])
            h, w = ann['height'], ann['width']

            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']
                masks.append(mask)

                bbox = xywh2xyxy(ann['annotation'][i]['bbox'])
                bboxes.append(bbox)
        
            for i in range(len(ann['conversations'])//2):
                    
                # Mask appears same order as conversation
                question = ann['conversations'][i*2]['value']
                qa_s.append({'from': 'human', 'value': question})         
             
                answer = ann['conversations'][i*2+1]['value']
                qa_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = img_path,
                bboxes = bboxes,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s
            ))
            # print(qa_s)

        return data_infos

    def __getitem__(self, i, debug=False):
        data_info = self.data_infos[i]
        img_path = data_info['img_path']
        height = data_info['height']
        width = data_info['width']
        bboxes = data_info['bboxes']
        masks_raw = data_info['masks']
        masks = []
        for mask_r in masks_raw:
            mask = self.annToMask(mask_r, height, width)
            masks.append(mask)
        ori_h, ori_w = masks[0].shape
            
        masks = np.array(masks)
        qas = data_info['qas']

        image, image_size, image_token_len = self.process_image(img_path)

        # Replace with bbox:
        bbox_texts = [self.bbox_to_text(bbox, ori_h, ori_w) for bbox in bboxes]
        for idx,qa in enumerate(qas):
            if idx % 2 == 1:
                continue
            question = qa['value']
            if self.use_bbox_text:
                # Replace <region{id}> tokens with corresponding bbox text
                # Convert 1-index in region text to 0-index when referring to bbox e.g. region1 -> <mask><pos> {bbox_text0}
                question = re.sub(r'<region(\d+)>', lambda match: f'<mask><pos> {bbox_texts[int(match.group(1))-1]}', question)
            else:
                question = re.sub(r'<region\d+>', '<mask><pos>', question)
            qa['value'] = question
        qas[0]['value'] = self.begin_str + qas[0]['value']

        sources = preprocess_multimodal(
            copy.deepcopy([qas]),
            self.data_args, image_token_len)
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

        data_dict['masks'] = torch.Tensor(masks)

        return data_dict

      

class OspreyConversations(ConversationDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 use_bbox_text=False,
                 ):
        self.limit = ""
        super().__init__(tokenizer, data_args, ann_file, img_prefix, use_bbox_text=use_bbox_text)

class OspreyShortForm(ConversationDataset):
     def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 use_bbox_text=False,
                 ):
        self.limit = ' Answer the question using a single word or phrase.'
        super().__init__(tokenizer, data_args, ann_file, img_prefix, use_bbox_text=use_bbox_text)

class OspreyDetailedDescription(ConversationDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 use_bbox_text=False,
                 ):
        super().__init__(tokenizer, data_args, ann_file, img_prefix, use_bbox_text=use_bbox_text)
    
    def get_question(self, bbox_text: str=None):
        """
        Question for detailed descrption.

        Args:
            bbox_texts (str, optional): bbox text to use. Defaults to None.

        Returns:
            str: A string of region references.
        """
        
        ref_string = '<mask><pos>' 
        if bbox_text:
            ref_string = f'<mask><pos> {bbox_text}'

        # ref_prefix = random.choice(REF_WAY)
        question = random.choice(DETAILED_QUESTIONS)
        question = question.replace('<region>', ref_string)

        return question

    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file))

        good = []
        for idx, ann in enumerate(ann_list):
            masks = []
            bboxes = []
            qa_s = []
            filename = ann['file_name'].split('_')[-1]
            img_path = os.path.join(self.img_prefix, filename)
            region_num = len(ann['annotation'])
            h, w = ann['height'], ann['width']
            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']
                masks.append(mask)

                bbox = xywh2xyxy(ann['annotation'][i]['bbox'])
                bboxes.append(bbox)

                bbox_text = self.bbox_to_text(bbox, h, w) if self.use_bbox_text else None
                question = self.get_question(bbox_text)
            
                qa_s.append({'from': 'human', 'value': question})     
                answer = re.findall(r"<.*>:\ (.*)", ann['description'][i])[0]
                qa_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = img_path,
                bboxes = bboxes,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s
            ))

            good.append(ann)
        return data_infos
    
    def __getitem__(self, i, debug=False):
        data_info = self.data_infos[i]
        img_path = data_info['img_path']
        height = data_info['height']
        width = data_info['width']
        bboxes = data_info['bboxes']
        masks_raw = data_info['masks']
        masks = []
        for mask_r in masks_raw:
            mask = self.annToMask(mask_r, height, width)
            masks.append(mask)
        
        assert len(bboxes) == len(masks)
            
        masks = np.array(masks)
        qas = data_info['qas']
        qas = copy.deepcopy(qas)

        image, image_size, image_token_len = self.process_image(img_path)

        # Add image prefix
        qas[0]['value'] = self.begin_str + qas[0]['value']

        sources = preprocess_multimodal(
            copy.deepcopy([qas]),
            self.data_args, image_token_len)
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

        data_dict['masks'] = torch.Tensor(masks)

        return data_dict

if __name__ == '__main__':
    from transformers import AutoTokenizer, CLIPImageProcessor
    from types import SimpleNamespace
    from osprey.model.multimodal_encoder.clip_encoder import get_image_processor
    tokenizer = AutoTokenizer.from_pretrained('../models/Osprey-7b/')
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')
    dataset = OspreyDetailedDescription(tokenizer, data_args, 
                                  ann_file='./data/Osprey-724K/osprey_detailed_description.json', 
                                  img_prefix='../images/coco/train2017', 
                                  use_bbox_text=True)
    breakpoint()
    from tqdm import tqdm
    for datum in tqdm(dataset):
        pass
    # dataset = OspreyLVISPosNeg(None,ann_file='./data/Osprey-724K/osprey_lvis_positive_negative.json', img_prefix='/dev/shm/train2017')
    # dataset = OspreyShortForm(None,ann_file='./data/Osprey-724K/osprey_short_form.json', img_prefix='/dev/shm/train2017')
    # dataset = OspreyPartLevel(None,ann_file='./data/Osprey-724K/osprey_part_level.json', img_prefix='/dev/shm/train2017')

