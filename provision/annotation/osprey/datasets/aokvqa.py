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

from .stage2_data import CustomDataset
from .cot_sg import TextCoTSGDataset
from .prompts import MC_QUESTIONS, SG_COT_QUESTIONS
from osprey.train.train import preprocess, preprocess_multimodal


BEGIN_STR = "<image>\nThis provides an overview of the picture and <region1> <mask><pos> highlighting the entire image.\n"

class AOKVQADataset(CustomDataset, TextCoTSGDataset):
    CLASSES = ('object',)

    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=20,
                 max_attributes_per_obj=5,
                 max_relations_per_obj=5,
                 is_train=True,
                 no_cot=False,
                 sample=None,
                 ):
        
        self.begin_str = BEGIN_STR
        self.no_cot = no_cot
        self.is_train = is_train
        self.max_attributes_per_obj = max_attributes_per_obj
        self.max_relations_per_obj = max_relations_per_obj
        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img, sample)

    def load_annotations(self, ann_file):

        letters = ['A', 'B', 'C', 'D']
        with open(ann_file, 'r') as f:
          data = json.load(f)
        data_infos = []

        for ann in tqdm(data):

            sg: dict = ann['scene_graph']
            image_id = str(ann['coco_id']).zfill(12)

            scene_graph_text = '\n'.join([f"{k}: {self.get_sg_dict(v)}" for k,v in sg.items()][:self.max_gt_per_img])
            img_path = os.path.join(self.img_prefix, image_id+'.jpg')

            choices: List[str] = ann['choices']
            choice_text: str = '\n'.join([f'{letters[idx]}. {c}' for idx, c in enumerate(choices)])

            if self.no_cot:
                q = random.choice(MC_QUESTIONS)
                output_text = f"Answer: {letters[ann['label']]}"
            else:
                q = random.choice(SG_COT_QUESTIONS)
                output_text = f"{scene_graph_text}\nAnswer: {letters[ann['label']]}"
            input_text = f"{q}\n{ann['question']}\n{choice_text}"
            sg_s = []
            sg_s.append({'from': 'human', 'value': input_text})
            sg_s.append({'from': 'gpt', 'value': output_text})

            data_infos.append(dict(
                img_path = img_path,
                scene_graph = sg,
                sgs = sg_s
            ))

        return data_infos
    
    def get_sg_dict(self, info: dict) -> dict:
        """ 
        Generates a scene graph dictionary from the provided information and object mapping.

        Args:
            info (dict): The information dictionary containing name, attributes, and relations.

            Returns:
                dict: A scene graph dictionary with the following structure:
                    {
                        'name': name, # Name of the scene
                        'attributes': attributes, # Attributes of the scene
                        'relations': relations, # Relations in the scene
                    }
                    Note: Keys with empty values are removed to free up tokenization space.
        """

        def parse_relations(relations):
            output = []
            for relation in relations:
                obj_name = relation['object']
                output.append(f"{relation['name']} {obj_name}")
            return output

        name: str = info['name']
        attributes: list = info.get('attributes', [])[:self.max_attributes_per_obj]
        relations = [rel for rel in info.get('relations', []) if isinstance(rel, dict)]
        relations = parse_relations(relations)
        if self.is_train:
            random.shuffle(relations)
            relations = relations[:self.max_relations_per_obj]
        sg_dict = {'name': name, 'bbox': info['bbox'], 'attributes': attributes, 'relations': relations}

        # remove key-values that are empty to free up tokenization space
        for k in list(sg_dict.keys()):
            if isinstance(sg_dict[k], list) and len(sg_dict[k]) == 0:
                sg_dict.pop(k)
        
        return sg_dict

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, i, debug=False):
        data_info = self.data_infos[i]

        img_path = data_info['img_path']
        sgs = data_info['sgs']
        processor = self.data_args.image_processor
        image = Image.open(img_path).convert('RGB')
        w, h = image.size

        pred_masks = self.get_mask(w=w, h=h)

        image = processor.preprocess(image,
                                     do_center_crop=False,
                                     return_tensors='pt')['pixel_values'][0]

        image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                size=(512, 512),
                                                mode='bilinear',
                                                align_corners=False).squeeze(0)

        cur_token_len = (image.shape[1] // 16) * (image.shape[2] // 16)  # FIXME: 16 is hardcoded patch size
        sgs = copy.deepcopy(sgs)
        sgs[0]['value'] = self.begin_str + sgs[0]['value']
        if debug:
            for sg in sgs:
                print(sg['from'])
                print(sg['value'])
                print()

        sources = preprocess_multimodal(
            copy.deepcopy([sgs]),
            self.data_args, cur_token_len)

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
    import cv2
    import supervision as sv
    tokenizer = AutoTokenizer.from_pretrained('models/Osprey-7b/')
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')
    dataset = AOKVQADataset(
        tokenizer, data_args=data_args, 
        ann_file="data/aokvqa/aokvqa_cogvlm_4bit_dense_gt_captions_grounding_objects_regions_scene_graph_gpt4_turbo_osprey_input.json", 
        img_prefix="/mmfs1/gscratch/raivn/jspark96/data/images/coco/train2017",
        no_cot=True,
        )
    data = dataset.__getitem__(0, debug=True)

    def draw_segmentation(idx: int):
        info =  dataset.data_infos[idx]
        sg = info['sgs']
        img_path = info['img_path']
        im = cv2.imread(img_path)
        h,w = im.shape[:2]
        scene_graph = info['scene_graph']
        labels = [v['name'] for v in scene_graph.values()]
        boxes = np.array([v['bbox'] for v in scene_graph.values()]) / 1000
        boxes[:,[0,2]] *= w
        boxes[:,[1,3]] *= h
        detections = sv.Detections(xyxy=boxes)
        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=im.copy(), detections=detections, labels=labels)
        cv2.imwrite('gqa_cot_sg.jpg',annotated_image)
    
    cur_input_ids = data['input_ids']
    mask_idx = torch.nonzero(cur_input_ids==tokenizer.convert_tokens_to_ids(['<mask>'])[0])
    assert len(mask_idx) == len(data['masks']), "mask num not equal to mask feats"
    draw_segmentation(0)
    breakpoint()
    bad_id = [] 
    for idx in tqdm(range(len(dataset))):
        try:
            dataset.__getitem__(idx)
        except Exception:
            bad_id.append(idx)
    print('bad ids: {}'.format(bad_id))
    breakpoint()
