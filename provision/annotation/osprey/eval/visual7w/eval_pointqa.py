import argparse
from pathlib import Path
import torch
import os
import json
from tqdm import tqdm
from pprint import pprint
import pandas as pd
import re
from PIL import Image

from typing import List, Literal, Dict, Tuple, Generator
from osprey.train.train import DataArguments
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.mm_utils import tokenizer_image_token
from osprey.datasets.refexp import RefExpDataset, GROUNDING_QUESTIONS
from osprey.eval.eval import OspreyEval

import cv2
import numpy as np

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

class PointQAEval(OspreyEval, RefExpDataset):
    ''' 
    RefExp evaluation
    '''

    def __init__(
            self, 
            model_path, 
            region_mode: Literal['box', 'segmentation', 'box_segmentation']='box',
            chunk_idx:int=0,
            num_chunks:int=1,
            debug=False, ):
        
        super().__init__(model_path, chunk_idx=chunk_idx, num_chunks=num_chunks, debug=debug)
        self.region_mode = region_mode
    
    def eval(self, root_path, ann_file, temperature=0.0, top_p=1.0):

        sgs = []
        ann = pd.read_json(ann_file, lines=True).to_dict(orient='records')

        data = self.get_chunk(ann)

        for idx, datum in enumerate(tqdm(data)):

            img_path = os.path.join(root_path, datum['file_path'])
            width, height = Image.open(img_path).size

            # Process regions
            boxes = datum['candidates']
            segs = boxes

            # Create masks and regions
            masks = self.create_masks(boxes, segs, height, width)
            masks = torch.from_numpy(masks)
            num_regions = len(segs)

            region_string = self.get_region_string(num_regions)
            prompt = region_string + ' ' + GROUNDING_QUESTIONS[0] + datum['question']
            print("Question: ", prompt)
            answer = 3

            # Adds begin_string
            init_inputs: dict = self.get_init_inputs(img_path,
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

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            # Generated relations per object
            outputs: str = self.get_outputs(image, input_ids, masks, stop_str, 
                                            temperature, top_p, max_new_tokens=10, do_sample=False)
            pred_answer = get_region_id(outputs)

            print("[GT]")
            print(answer)
            print("[Pred]")
            print(outputs, pred_answer)

            sgs.append({
                'image_id': datum['genome_id'],
                'question_id': datum['qa_id'], 
                'question': datum['question'],
                'answer': answer,
                'output': outputs,
                'pred_answer': pred_answer,
            })

        return sgs

def get_region_id(outputs: str) -> int:
    ''' 
    Use regex to find the first instance of region{id} and return the id. 
    Subtract by 1 to get the 0-index
    '''
    match = re.search(r'region(\d+)', outputs)
    if match:
        return int(match.group(1)) - 1
    else:
        print('could not get region id from {}'.format(outputs))
        return 0


if __name__ == '__main__':
    '''
        python -m osprey.eval.visual7w.eval_pointqa --model exp/multi_region_v3_cot_bs16/ \
            --temperature 0.01 \
            --top_p 1.0 \
            --output osprey/eval/results/pointqa/multi_region_v3_cot_bs16/temp0.01.jsonl
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument('--json', help='path to gqa json file with regions', 
                        default='/net/nfs.cirrascale/prior/jamesp/data/visual7w/v7w_pointing_val.jsonl')
    parser.add_argument('--img', help='path to COCO imgs', default='../images/visual7w/images/')
    parser.add_argument("--region_mode", type=str, default='box')
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--debug", action='store_true', help='Debug mode')

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument('--log_wandb', action='store_true', help='Log results to wandb')
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--wandb_key", type=str)
    args = parser.parse_args()

    refexp_eval = PointQAEval(args.model, region_mode=args.region_mode,
                        chunk_idx=args.chunk_idx, num_chunks=args.num_chunks, debug=args.debug)
    
    results = refexp_eval.eval(args.img, args.json, args.temperature, args.top_p)
    
    os.makedirs(Path(args.output).parent, exist_ok=True)
    df =  pd.DataFrame(results)
    print('Acc', (df['answer'] == df['pred_answer']).mean())
    df.to_json(args.output, orient='records', lines=True)
    print('Saved results to {}'.format(args.output))

    if args.log_wandb:
        OspreyEval.log_wandb(args.wandb_run_name, args.wandb_key, {args.wandb_key: 0.0})