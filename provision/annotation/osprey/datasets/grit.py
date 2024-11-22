import numpy as np
import pandas as pd
import torch
import json
import os
import copy
from tqdm import tqdm
from .llava import LlavaDataset
from .prompts import GROUNDING_QUESTIONS
from braceexpand import braceexpand

def get_whole_image_mask(height, width) -> np.ndarray:
    mask = np.ones((height, width), dtype=np.bool_)
    return mask.astype(np.float32)

class GritDataset(LlavaDataset):

    def process_llava_text(self, s: str):
        s = s.replace('<image>\n','')
        return s
    
    def load_annotations(self, ann_file):

        data_infos = []
        parquet_files = braceexpand(ann_file)

        # Load list of parquet files
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            ann_list = df.to_dict(orient='records')
            for ann in tqdm(ann_list):

                qa_s = []
                img_path = os.path.join(self.img_prefix, str(ann['id'])+'.jpg')
                caption = ann['caption']
                for ck in ann['noun_chunks']:
                    noun = caption[int(ck[0]):int(ck[1])]
                    norm_bbox = ck[2:6]
                    bbox_text = self.textify_bbox(norm_bbox)

                    question = np.random.choice(GROUNDING_QUESTIONS) + noun
                    qa_s.append({'from': 'human', 'value': question})         
                    qa_s.append({'from': 'gpt', 'value': bbox_text})

                data_infos.append(dict(
                    img_path = img_path,
                    convs = qa_s
                ))

        return data_infos

if __name__ == '__main__':
    from transformers import AutoTokenizer, CLIPImageProcessor
    from types import SimpleNamespace
    from osprey.model.multimodal_encoder.clip_encoder import get_image_processor
    tokenizer = AutoTokenizer.from_pretrained('../models/Osprey-7b/')
    tokenizer = AutoTokenizer.from_pretrained('../models/mistralai/Mistral-7B-Instruct-v0.2/')
    tokenizer.add_tokens(['<mask>','<pos>'], special_tokens=True)
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')
    dataset = GritDataset(
        tokenizer, 
        data_args=data_args, 
        ann_file= "/net/nfs.cirrascale/mosaic/jamesp/data/GRIT/grit-20m/coyo_0_snappy.parquet",
        img_prefix="/net/nfs.cirrascale/mosaic/jamesp/data/GRIT/grit-20m/coyo_0_snappy",
    )
    data_info = dataset.get_data_info(0)
    data = dataset.__getitem__(0, debug=True)
    breakpoint()
