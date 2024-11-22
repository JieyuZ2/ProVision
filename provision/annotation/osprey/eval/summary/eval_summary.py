import argparse
from pathlib import Path
import torch
import os
import json
from tqdm import tqdm
import pandas as pd
import re
from functools import partial
from typing import List, Literal, Dict, Tuple
from transformers import AutoTokenizer, CLIPImageProcessor
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.mm_utils import tokenizer_image_token
from osprey.train.train import preprocess_multimodal
from osprey.train.train import DataArguments
from osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM

from osprey.datasets.relation_category import RelationSummaryDataset, RELATION_SUMMARY_QUESTIONS, Ref_WAY
from osprey.eval.eval import OspreyEval

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import argparse

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True


class SummaryEval(OspreyEval, RelationSummaryDataset):
    ''' 
    '''
    def __init__(
            self, 
            model_path, 
            max_regions=150,
            region_mode: Literal['box', 'segmentation', 'box_segmentation']='segmentation',
            chunk_idx:int=0,
            num_chunks:int=1,
            debug=False, ):
        
        super().__init__(model_path, chunk_idx=chunk_idx, num_chunks=num_chunks, debug=debug)
        self.max_regions = max_regions

        self.region_mode = region_mode
        self.is_train = False

    def eval(self, root_path, ann_file, temperature=0.2, top_p=1.0):

        sgs = []
        anns = pd.read_json(ann_file, lines=True).to_dict(orient='records')
        anns = anns.to_dict(orient='records')

        anns = self.get_chunk(anns)

        for ann in tqdm(anns[:100]):
            image_id = str(ann['image_id'])
            height = ann['height']
            width = ann['width']
            regions: list = ann['regions'][:self.max_regions]

            # Gather regions
            segs = []
            boxes = []
            for region in regions:
                # add region segmentations
                boxes.append(region['bbox'])
                segs.append(region['segmentation'])
            
            img_path = os.path.join(root_path, image_id+'.jpg')

            # Create masks and regions
            masks = self.create_masks(boxes, segs, height, width)
            masks = torch.from_numpy(masks)

            # Generate summary
            pred_summary: str = self.generate(img_path=img_path, masks=masks, temperature=temperature, top_p=top_p)

            print("[Pred]")
            print(pred_summary)

            # Parse prediction
            # pred_summary = self.parse_summary(outputs) 

            sgs.append({
                'image_id': image_id,
                'bboxes': boxes,
                'pred_summary': pred_summary,
                'width': width, 
                'height': height
            })

        return sgs
    
    def generate(self, img_path, masks, temperature=1.0, top_p=1.0):

        # Create input IDs
        num_objects = len(masks)
        prompt = self.get_region_prompt(num_objects)
        init_inputs = self.get_init_inputs(img_path,
                                    self.image_processor,
                                    masks=masks,
                                    prompt=prompt,
                                    )

        image = init_inputs['image']
        masks = init_inputs['masks'].cuda()

        conv = self.get_new_conv()
        qs = init_inputs['sources'][0][0]['value']

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Input IDs with image token
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        # Generate Summary
        pred_summary: str = self.get_outputs(image, input_ids, masks, stop_str, temperature, top_p, max_new_tokens=1024)

        return pred_summary

    def get_region_prompt(self, num_objects: int) -> str:
        ref_string = ''
        for i in range(num_objects):
            ref_string = ref_string +  f'region{i+1} <mask><pos>' + ','
        ref_string = ref_string[:-1]
        ref_prefix = Ref_WAY[0]

        begin_string = ref_prefix.replace('<region>', ref_string)

        question = self.get_question()

        return begin_string + ' ' + question
    
    def get_question(self):
        return RELATION_SUMMARY_QUESTIONS[0]
    
    def parse_summary(self, summary: str) -> str:
        '''
        Parse the generated summary to get the predicted regions.
        '''
        pred_regions = []
        for region in summary.split(' '):
            if region.startswith('region'):
                pred_regions.append(int(region.split('region')[1].split('<')[0]))
        return pred_regions
     
if __name__ == "__main__":
    '''
        python -m osprey.eval.summary.eval_summary --model exp/relation_description_summary_coco_sam_seem_box_segm/checkpoint-6000/ \
            --json data/sg/vg_test_sg_sam_hq.jsonl \
            --temperature 0.5 \
            --top_p 0.95 \
            --output osprey/eval/results/summary/test/relation_description_summary_coco_sam_seem-gt_objects_temp0.5_top0.95.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey sg evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--bert', help='path to bert model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--json', help='path to gqa json file with regions', default='data/sg/vg_test_sg_sam_hq_subset.jsonl')# default='data/sg/test_vg_sg_sam_hq.json')
    parser.add_argument('--img', help='path to gqa imgs', default='/mmfs1/gscratch/raivn/jspark96/data/images/gqa/images')
    parser.add_argument('--region_mode', type=str, default='segmentation', choices=['box', 'segmentation', 'box_segmentation'])
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_gt_objects', action='store_true', help='used gt objects when generating relations for sg_mode: 3')
    args = parser.parse_args()

    relation_eval = SummaryEval(args.model, 
                     debug=args.debug,
                     region_mode=args.region_mode,
                    )
    results = relation_eval.eval(args.img, args.json, args.temperature)
    os.makedirs(Path(args.output).parent, exist_ok=True)
    pd.DataFrame(results).to_json(args.output, orient='records', lines=True)

