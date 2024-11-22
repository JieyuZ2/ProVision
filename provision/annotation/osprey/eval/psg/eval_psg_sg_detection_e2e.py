import argparse
from pathlib import Path
import torch
import os
import json
import re
import random
import numpy as np
import logging
from tqdm import tqdm
from pprint import pprint
import pandas as pd
from typing import List, Dict, Tuple
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.mm_utils import tokenizer_image_token
from osprey.train.train import DataArguments

from osprey.datasets.sg_detection import REGION_PROPOSAL_QUESTIONS, SG_QUESTIONS
from osprey.eval.psg.eval_psg import PSGEval

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

class PSGFullSGDetectionEval(PSGEval):
    ''' 
    Scene Graph Detection evaluation: Predict object labels and relationships with detected regions.
    Regions with High IoUs with Reference regions are considered in evaluation as candidates.
    '''

    def get_proposed_boxes(self, region_outputs, height, width):
        # Regular expression to find all bounding box regions within double square brackets
        bbox_pattern = r'\[\[(.*?)\]\]'

        # Find all matches
        bbox_strings = re.findall(bbox_pattern, region_outputs)

        # Convert the extracted string regions to lists of integers
        bbox_regions = [list(map(int, bbox.split(','))) for bbox in bbox_strings]        

        # Unnormalize the bounding box regions
        boxes = []
        for bbox in bbox_regions:
            bbox[0] = int(bbox[0]/1000 * width)
            bbox[1] = int(bbox[1]/1000 * height)
            bbox[2] = int(bbox[2]/1000 * width)
            bbox[3] = int(bbox[3]/1000 * height)
            boxes.append(bbox)
        return boxes
    
    def create_masks_from_boxes(self, boxes, height, width):
        return np.array([self.bboxToMask(box, height, width) for box in boxes])

    def eval(self, root_path, pan_seg_path, ann_file, use_bbox_text=False, 
             temperature=0.2, top_p=1.0):

        sgs = []
        ann = json.load(open(ann_file))

        data = self.get_chunk(ann['data'])

        for idx, datum in enumerate(tqdm(data)):

            img_path = os.path.join(root_path, datum['file_name'])
            image_id = str(datum['image_id'])
            height = datum['height']
            width = datum['width']

            ''' Get region proposals '''
            question = REGION_PROPOSAL_QUESTIONS[0]
            init_inputs = self.get_init_inputs(
                img_path,
                self.image_processor,
                prompt=question,
                masks=None,
            )
            image = init_inputs['image']
            masks = None

            conv = self.get_new_conv()
            region_qs = init_inputs['sources'][0][0]['value']
            if idx == 0:
                print(region_qs)

            conv.append_message(conv.roles[0], region_qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Get Proposed Regions
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            region_outputs: str = self.get_outputs(image, input_ids, None, self.stop_str, 
                                                temperature, top_p, max_new_tokens=1024, do_sample=True)
            # unnormalize bounding boxes
            pred_boxes = self.get_proposed_boxes(region_outputs, height, width)
            pred_boxes = pred_boxes[:self.max_regions]

            ''' Get scene graph from proposed regions'''
            question = SG_QUESTIONS[0]
            masks = self.create_masks_from_boxes(pred_boxes, height, width)
            masks = torch.from_numpy(masks)
            num_objects = len(masks)
            region_string = self.get_region_string(n=num_objects, bbox_texts=None)
            prompt = region_string + ' ' + question
            init_inputs = self.get_init_inputs(
                img_path,
                self.image_processor,
                prompt=prompt,
                masks=masks,
            )
            image = init_inputs['image']
            masks = init_inputs['masks'].cuda()

            conv = self.get_new_conv()
            qs = init_inputs['sources'][0][0]['value']
            conv.append_message(conv.roles[0], region_qs)
            conv.append_message(conv.roles[1], region_outputs)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            sg_outputs: str = self.get_outputs(image, input_ids, masks, self.stop_str, 
                                                temperature, top_p, max_new_tokens=1024, do_sample=True)
    
            if idx < 20:
                print("[Pred]")
                print(pred_boxes)
                print(sg_outputs)

                print("[GT]")
                print(datum['relations'])

            gt_boxes = np.array([a['bbox'] for a in datum['annotations']])
            gt_object_labels = [d['category_id'] for d in datum['annotations']]
            sgs.append({
                'image_id': image_id,
                'question_id': datum['question_id'],
                'width': width, 
                'height': height,
                'gt_labels': datum['relations'],
                'gt_boxes': gt_boxes,
                'gt_object_labels': gt_object_labels,
                'pred_boxes': pred_boxes,
                'pred_raw': sg_outputs,

                # 'pred_names': prediction_names,
                # 'pred_labels': prediction_labels,
                # 'pred_object_names': object_outputs,
                # 'pred_object_labels': object_labels,
            })

            if idx == 0:
                pprint(sgs[0])
                # break
            
        return sgs

if __name__ == "__main__":
    '''
        python -m osprey.eval.psg.eval_psg_sg_detection_e2e \
            --model exp/osprey_sg_vg_psg_merged_full_v4_mode_2_box-osprey_stage_3-mistral-7b-instruct-v0.2_bs16_epoch2/checkpoint-4372/ \
            --max_regions 99 \
            --temperature 0.5 \
            --top_p 0.95 \
            --region_mode box \
            --output osprey/eval/results/psg/psg_sg_detection_e2e/osprey_sg_vg_psg_merged_full_v4_mode_2_box-mistral-7b/temp0.5_top0.95_max_regions_99.jsonl
        
        python -m osprey.eval.psg.eval_psg_sg_detection_e2e \
            --model /net/nfs.cirrascale/prior/jamesp/Osprey/exp/osprey_sg_vg_psg_merged_full_v4_mode_0_box-osprey_stage_3-mistral-7b-instruct-v0.2-lr2e-5_bs16_epoch3/ \
            --max_regions 99 \
            --temperature 0.5 \
            --top_p 0.95 \
            --region_mode box \
            --output osprey/eval/results/psg/psg_sg_detection_e2e/osprey_sg_vg_psg_merged_full_v4_mode_0_box-osprey_stage_3-mistral-7b-instruct-v0.2-lr2e-5_bs16_epoch3/temp0.5_top0.95_max_regions_99.jsonl
        
        python -m osprey.eval.psg.eval_psg_sg_detection_e2e \
            --model /net/nfs.cirrascale/prior/jamesp/Osprey/exp/multi_region_v10_gqa_cot_osprey_stage3_llava_pointqa_e2e_sg_bbox_text-lr1e-5_bs32_epoch3_v2/checkpoint-89703/ \
            --max_regions 99 \
            --temperature 0.5 \
            --top_p 0.95 \
            --region_mode box \
            --output osprey/eval/results/psg/psg_sg_detection_e2e/multi_region_v10_gqa_cot_osprey_stage3_llava_pointqa_e2e_sg_bbox_text-lr1e-5_bs32_epoch3/checkpoint-89703/temp0.5_top0.95_max_regions_99.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey sg evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--bert', help='path to bert model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--category', help='path to PSG Categories', default='osprey/eval/psg/psg_category.json')
    parser.add_argument('--json', help='path to psg file with detected regions', 
                        default='osprey/eval/psg/psg_asv2_val_test_regions.json')
    parser.add_argument('--img', help='path to gqa imgs', default='../images/coco')
    parser.add_argument('--pan_seg_img', help='path to gqa imgs', default='../data/coco/panoptic_annotations')

    # Region config
    parser.add_argument('--max_regions', type=int, default=30)
    parser.add_argument('--region_mode', type=str, default='segmentation', choices=['box', 'segmentation', 'box_segmentation'])
    parser.add_argument('--description_context', action='store_true', help='Use previously generated description as context to generate relations')
    parser.add_argument('--full_sg', action='store_true', help='Generate full scene graph end to end')
    parser.add_argument('--use_bbox_text', action='store_true', help='Generate full scene graph end to end')

    # Gen config
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    print('Saving results to {}'.format(args.output))
    eval_class = PSGFullSGDetectionEval
    sg_eval = eval_class(args.model, 
                    max_regions=args.max_regions,        
                    bert_model=args.bert,
                    category=args.category,
                    debug=args.debug,
                    region_mode=args.region_mode,
                    chunk_idx=args.chunk_idx,
                    num_chunks=args.num_chunks,
    )
    results = sg_eval.eval(args.img, args.pan_seg_img, args.json, args.use_bbox_text,
                                 args.temperature, args.top_p)

    os.makedirs(Path(args.output).parent, exist_ok=True)
    pd.DataFrame(results).to_json(args.output, orient='records', lines=True)
    print('Saved results to {}'.format(args.output))
 

