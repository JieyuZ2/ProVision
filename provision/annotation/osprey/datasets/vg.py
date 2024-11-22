import copy
import random
import os
import numpy as np
import torch
from .stage2_data import CustomDataset, xywh2xyxy
from osprey.train.train import preprocess, preprocess_multimodal

LIMIT = " Answer the question using a short phrase."
QUESTIONS =  [
    'Give me a short description of <region>.',
    'Can you give me a short description of <region>?',
    'Can you provide me with a short description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in few words?",
    'What can you tell me about the region indicated by <region> in the image in few words?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a concise description?",
    'Could you describe the region shown as <region> in the picture concisely?',
    'What can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a brief description of the region marked with <region> in the image.',
    'Can you give me a brief introduction of the region labeled as <region> in the picture?',
    "I'm interested in knowing the region represented by <region> in the photo. Can you describe it in several words?",
    'What is the region outlined by <region> in the picture like? Could you give me a streamlined description?',
    'Can you provide me with a brief description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in few words, please?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a simple description?",
    'Could you describe the region shown as <region> in the picture in several words?',
    'Please provide me with a simple description of the region marked with <region> in the image, please.',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in few words, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a simple and clear description?',
    'Please describe the region <region> in the image concisely.',
    'Can you offer a simple analysis of the region <region> in the image?',
    'Could tell me something about the region highlighted by <region> in the picture briefly?',
    'Can you share a simple rundown of the region denoted by <region> in the presented image?'
]

class VGDATA(CustomDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=3,
                 use_bbox_text=False,
                 ):

        self.data_args = data_args
        self.tokenizer = tokenizer
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.max_gt_per_img = max_gt_per_img

        self.begin_str = """<image>\nThis provides an overview of the picture.\n"""
        
        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img,
                         use_bbox_text=use_bbox_text)


    def get_data_item(self, idx):
        data_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)

        img_path = os.path.join(self.img_prefix, data_info['filename'])
        image, image_size, image_token_len = self.process_image(img_path)

        gt_bboxes = []
        gt_masks = []
        gt_labels = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            mask = self.annToMask(ann['segmentation'], data_info['height'], data_info['width'])
            
            gt_labels.append(ann['caption'])
            gt_bboxes.append(xywh2xyxy(ann['bbox']))
            gt_masks.append(mask)


        data_item = dict(
            image = image,
            image_token_len=image_token_len,
            gt_bboxes=gt_bboxes,
            gt_masks=gt_masks,
            gt_labels=gt_labels,
        )
        return data_item

    
    def process_text(self, data_item, debug=False):
        image = data_item['image']
        image_token_len = data_item['image_token_len']
        ori_labels = data_item['gt_labels']
        ori_masks = np.array(data_item['gt_masks'])
        ori_masks = torch.from_numpy(ori_masks) 

        ori_bboxes = data_item['gt_bboxes']

        shuffle_ids = torch.randperm(len(ori_labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        ori_masks = ori_masks[shuffle_ids]
        ori_labels = [ori_labels[i] for i in shuffle_ids]
        ori_bboxes = [ori_bboxes[i] for i in shuffle_ids]
        ori_h, ori_w = ori_masks[0].shape

        sources = dict()

        sources['conversations'] = []

        for i in range(len(ori_labels)):
            question = random.choice(QUESTIONS).strip()
            if self.use_bbox_text:
                bbox_text = self.bbox_to_text(ori_bboxes[i], ori_h, ori_w)
                question = question.replace('<region>', f'<mask><pos> {bbox_text}')
            else:
                question = question.replace('<region>', '<mask><pos>')
            if i == 0:
                question = self.begin_str + question
            question += LIMIT
            answer = ori_labels[i]
            sources['conversations'].append(
                {'from': 'human', 'value': question})
            sources['conversations'].append({'from': 'gpt', 'value': answer})

        sources = preprocess_multimodal(
            copy.deepcopy([sources['conversations']]),
            self.data_args,
            image_token_len
        )
        if debug:
            for conv in sources[0]:
                print(conv['from'])
                print(conv['value'])
                print("=")

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True
        )
        
        # get single
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict['input_ids'][0],
                             labels=data_dict['labels'][0])

        data_dict['image'] = image
        data_dict['masks'] = ori_masks
        return data_dict

if __name__ == '__main__':
    from transformers import AutoTokenizer, CLIPImageProcessor
    from types import SimpleNamespace
    from osprey.model.multimodal_encoder.clip_encoder import get_image_processor
    tokenizer = AutoTokenizer.from_pretrained('../models/Osprey-7b/')
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')
    dataset = VGDATA(
        tokenizer, 
        data_args=data_args, 
        ann_file="./data/vg/vg_train_with_mask.json", 
        img_prefix="../images/vg/VG_100K_all",
        use_bbox_text=True
    )

    data_item = dataset.get_data_item(0)
    breakpoint()


