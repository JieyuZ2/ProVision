#!/usr/bin/env python
# coding: utf-8

# In[123]:

import json
from typing import List, Dict
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import torch

from torchvision.ops import box_iou
from osprey.eval.eval import OspreyEval

tqdm.pandas()

def eval_grounding(preds):
    res = defaultdict(list)
    for pred in preds:
        question_id = pred['question_id']

        gold_box = pred['gt_bbox']
        gold_box = torch.tensor(gold_box, dtype=torch.float32).view(-1, 4)

        pred_box = pred['pred_bbox']
        pred_box = torch.tensor(pred_box, dtype=torch.float32).view(-1, 4)
        iou = box_iou(pred_box, gold_box)
        iou = iou.item()

        res[pred['category']].append(iou > 0.5)
        res['total'].append(iou > 0.5)

    for k, v in sorted(res.items()):
        print(f'Category: {k}, # samples: {len(v)}, # Acc@0.5: {sum(v) / len(v) * 100:.2f}')
    

if __name__ == '__main__':
    '''
    python osprey/eval/refexp/run_refexp_eval.py \
        osprey/eval/results/refexp/multi_region_v1/max_regions_30_segmentation.jsonl
    
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", type=str)

    # wandb logs
    parser.add_argument('--log_wandb', action='store_true', help='Log results to wandb')
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--wandb_key", type=str)
    args = parser.parse_args()

    df = pd.read_json(args.result_file, lines=True)
    result: List[Dict] = df.to_dict(orient='records')
    
    df = df[~df['region_index'].isnull()]
    df['region_index'] = df['region_index'].apply(int)
    print('RefExp accuracy: {}'.format((df['region_index'] == df['pred_region_index']).mean()))

    eval_grounding(result)
    
    # breakpoint()

    # if args.log_wandb:
    #     OspreyEval.log_wandb(args.wandb_run_name, args.wandb_key, recalls)

