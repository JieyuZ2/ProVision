import numpy as np
import torch
import json
import os
import copy
from tqdm import tqdm
from .llava import LlavaDataset
from osprey.train.train import preprocess, preprocess_multimodal
from PIL import Image
from braceexpand import braceexpand

def get_whole_image_mask(height, width) -> np.ndarray:
    mask = np.ones((height, width), dtype=np.bool_)
    return mask.astype(np.float32)

class AllSeeingDataset(LlavaDataset):

    def process_llava_text(self, s: str):
        s = s.replace('<image>\n','')
        return s
    
    def load_annotations(self, ann_file):

        if os.path.isfile(ann_file):
            return super().load_annotations(ann_file)
        else:
            data_infos = []
            jsonl_files = braceexpand(ann_file)

            # Load list of jsonl files
            for jsonl_file in jsonl_files:
                with open(jsonl_file) as f:
                    for line in tqdm(f):
                        ann = json.loads(line)
                    
                        if len(ann['conversations'])//2 ==0:
                            continue
                        qa_s = []

                        if 'image' in ann:
                            filename = ann['image']
                            img_path = os.path.join(self.img_prefix, filename)
                            region_str = self.region_str
                        else:
                            img_path = None   
                            region_str = self.blank_str

                        for i in range(len(ann['conversations'])//2):
                                
                            question = ann['conversations'][i*2]['value']
                            question = self.process_llava_text(question)
                        
                            qa_s.append({'from': 'human', 'value': question})         

                            answer = ann['conversations'][i*2+1]['value']
                            answer = self.process_llava_text(answer)
                            qa_s.append({'from': 'gpt', 'value': answer})

                        data_infos.append(dict(
                            img_path = img_path,
                            convs = qa_s
                        ))

        return data_infos
    
    def visualize_bbox(self, image_path, bbox):
        image = Image.open(image_path)
        image_rgb = np.asarray(Image)
        detections = sv.Detections(bbox)
        visualize_masks(image_rgb, )
        return image

if __name__ == '__main__':
    from transformers import AutoTokenizer, CLIPImageProcessor
    from types import SimpleNamespace

    from osprey.eval.draw_utils import visualize_masks
    from osprey.model.multimodal_encoder.clip_encoder import get_image_processor
    import supervision as sv
    tokenizer = AutoTokenizer.from_pretrained('../models/Osprey-7b/')
    tokenizer = AutoTokenizer.from_pretrained('../models/mistralai/Mistral-7B-Instruct-v0.2/')
    tokenizer.add_tokens(['<mask>','<pos>'], special_tokens=True)
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')
    dataset = AllSeeingDataset(
        tokenizer, 
        data_args=data_args, 
        ann_file= "/net/nfs.cirrascale/mosaic/jamesp/code/all-seeing/all-seeing-v2/data/as_pretrain_10m/sa_{000020..000021}.jsonl",
        # ann_file= "/net/nfs.cirrascale/mosaic/jamesp/code/all-seeing/all-seeing-v2/data/as_pretrain_10m/sa_{000000..000255}.jsonl",
        img_prefix="/net/nfs.cirrascale/mosaic/jamesp/images/sam/",
    )
    data_info = dataset.get_data_info(0)
    data = dataset.__getitem__(0, debug=True)

    # Visualize bounding boxes with unnormalized image s
    breakpoint()
