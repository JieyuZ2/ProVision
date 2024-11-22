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
from osprey.train.train import DataArguments

from osprey.datasets.relation_category import RelationSummaryDataset, RELATION_SUMMARY_QUESTIONS, Ref_WAY
from osprey.eval.psg.eval_summary import SummaryEval
from osprey.eval.psg.eval_psg import create_segmentations

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import argparse

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

class PSGSummaryEval(SummaryEval):
    ''' 
    Scene Graph evaluation class that assigns generation result to scene graph object and relationship class
    using BertModel.
    '''

    def eval(self, root_path, pan_seg_path, ann_file, temperature=0.2, top_p=1.0):

        sgs = []
        ann = json.load(open(ann_file))

        classes = ann['classes']
        predicate_classes = ann['predicate_classes']

        data = self.get_chunk(ann['data'])

        for datum in tqdm(data):
            image_id = str(datum['image_id'])
            height = datum['height']
            width = datum['width']

            # Gather regions
            boxes = np.array([a['bbox'] for a in datum['annotations']])
            pan_seg_image = os.path.join(pan_seg_path, datum['pan_seg_file_name'])
            segments_info = datum['segments_info']
            segs: np.ndarray = create_segmentations(pan_seg_image, segments_info)
            
            img_path = os.path.join(root_path, datum['file_name'])

            # Create masks and regions
            masks = self.create_masks(boxes, segs, height, width)
            masks = torch.from_numpy(segs)

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
                'relations': datum['relations'],
                'relation_names': datum['relation_names'],
                'question_id': datum['question_id'],
                'width': width, 
                'height': height
            })

        return sgs

if __name__ == "__main__":
    '''
        python -m osprey.eval.psg.eval_psg_summary --model exp/relation_description_summary_coco_sam_seem_box_segm/checkpoint-6000/ \
            --temperature 0.5 \
            --top_p 0.95 \
            --output osprey/eval/results/summary/test/relation_description_summary_coco_sam_seem-gt_objects_temp0.5_top0.95.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey sg evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--bert', help='path to bert model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--json', help='path to gqa json file with regions', default='osprey/eval/psg/psg_asv2_val_test.json')# default='data/sg/test_vg_sg_sam_hq.json')
    parser.add_argument('--img', help='path to gqa imgs', default='../images/coco')
    parser.add_argument('--pan_seg_img', help='path to gqa imgs', default='../data/coco/panoptic_annotations')
    parser.add_argument('--region_mode', type=str, default='segmentation', choices=['box', 'segmentation', 'box_segmentation'])
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print('Saving results to {}'.format(args.output))
    relation_eval = PSGSummaryEval(args.model, 
                     debug=args.debug,
                     region_mode=args.region_mode,
                     chunk_idx=args.chunk_idx,
                     num_chunks=args.num_chunks,
                    )
    results = relation_eval.eval(args.img, args.pan_seg_img, args.json, args.temperature, args.top_p)
    os.makedirs(Path(args.output).parent, exist_ok=True)
    pd.DataFrame(results).to_json(args.output, orient='records', lines=True)
    print('Saved results to {}'.format(args.output))

