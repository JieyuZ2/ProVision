import numpy as np
import torch
import json
import os
import copy
from tqdm import tqdm
from .stage2_data import CustomDataset, xywh2xyxy
from osprey.train.train import preprocess, preprocess_multimodal
from PIL import Image

def get_whole_image_mask(height, width) -> np.ndarray:
    mask = np.ones((height, width), dtype=np.bool_)
    return mask.astype(np.float32)

class LlavaDataset(CustomDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 use_bbox_text=False,
                 sample=None,
                 ):
        self.begin_str = "<image>\nThis provides an overview of the picture.\n"
        # if use_bbox_text:
        #     whole_bbox = "[[0,0,1000,1000]]"
        #     self.region_str = f"There is region1 <mask><pos> {whole_bbox} for the entire image.\n"
        #     self.blank_str = f"There is region1 <mask><pos> {whole_bbox} for the blank image.\n"
        # else:
        #     self.region_str = "There is region1 <mask><pos> for the entire image.\n"
        #     self.blank_str = "There is region1 <mask><pos> for the blank image.\n"
        super().__init__(tokenizer, data_args, ann_file, img_prefix, use_bbox_text=use_bbox_text, sample=sample)
    
    def process_llava_text(self, s: str):
        s = s.replace('<image>\n','')
        s = s.replace('\n<image>','')
        s = s.replace('<','').replace('>','')
        return s
    
    def load_file(self, ann_file) -> list:
        # json or jsonl file
        if ann_file.endswith('.jsonl'):
            data = []
            with open(ann_file) as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        else:
            return json.load(open(ann_file))

    def load_annotations(self, ann_file):

        data_infos = []
        
        ann_list = self.load_file(ann_file)

        for ann in tqdm(ann_list):
            if len(ann['conversations'])//2 ==0:
                continue
            qa_s = []

            if 'image' in ann:
                filename = ann['image']
                img_path = os.path.join(self.img_prefix, filename)
                # region_str = self.region_str
            else:
                img_path = None   
                # region_str = self.blank_str

            for i in range(len(ann['conversations'])//2):
                    
                question = ann['conversations'][i*2]['value']
                question = self.process_llava_text(question)
                # if i==0:
                #     question = region_str + question
                qa_s.append({'from': 'human', 'value': question})         
                answer = ann['conversations'][i*2+1]['value']
                answer = self.process_llava_text(answer)
                qa_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = img_path,
                convs = qa_s
            ))

        return data_infos

    def __getitem__(self, i, debug=False):
        data_info = self.data_infos[i]
        img_path = data_info['img_path']
            
        # Load Image
        if img_path is not None:
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.new('RGB', (384,384))

        image, image_size, image_token_len = self.process_image(image)

        # process conversation
        convs = copy.deepcopy(data_info['convs'])
        convs[0]['value'] = self.begin_str + convs[0]['value']
        if debug:
            for conv in convs:
                print(conv['from'])
                print(conv['value'])
                print("=")
        sources = preprocess_multimodal(
            copy.deepcopy([convs]),
            self.data_args, 
            image_token_len
        )
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict['input_ids'][0],
                             labels=data_dict['labels'][0])

        data_dict['image'] = image
        data_dict['masks'] = None

        return data_dict


if __name__ == '__main__':
    from transformers import AutoTokenizer, CLIPImageProcessor
    from types import SimpleNamespace
    from osprey.model.multimodal_encoder.clip_encoder import get_image_processor
    tokenizer = AutoTokenizer.from_pretrained('../models/Osprey-7b/')
    tokenizer = AutoTokenizer.from_pretrained('../models/mistralai/Mistral-7B-Instruct-v0.2/')
    tokenizer.add_tokens(['<mask>','<pos>'], special_tokens=True)
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')

    dataset = LlavaDataset(
        tokenizer,
        data_args=data_args,
        ann_file="/net/nfs.cirrascale/mosaic/jamesp/images/llava/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json",
        img_prefix="/net/nfs.cirrascale/mosaic/jamesp/images/llava/LLaVA-Pretrain/images"
    )
    breakpoint()
    dataset = LlavaDataset(
        tokenizer, 
        data_args=data_args, 
        ann_file= "../data/LLaVAR/llavar_20k.json", 
        img_prefix="../data/LLaVAR/images",
    )

    dataset = LlavaDataset(
        tokenizer, 
        data_args=data_args, 
        ann_file= "../data/ShareGPT-4o/gpt-4o.jsonl", 
        img_prefix="../data/ShareGPT-4o/images"
    )
    
    dataset = LlavaDataset(
        tokenizer, 
        data_args=data_args, 
        ann_file= "./data/llava/llava_v1_5_sharegpt_40k.json", # "./data/llava/llava_v1_5_no_region_522k.json", 
        img_prefix="../images/",
    )
    data_info = dataset.get_data_info(0)
    data = dataset.__getitem__(0, debug=True)

    dataset = LlavaDataset(
        tokenizer, 
        data_args=data_args, 
        ann_file= "./data/llava/llava_instruct_150k.json", # "./data/llava/llava_v1_5_no_region_522k.json", 
        img_prefix="../images/coco/train2017/",
    )
    data_info = dataset.get_data_info(0)
    data = dataset.__getitem__(0, debug=True)

    breakpoint()

