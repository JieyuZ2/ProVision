import argparse
from pathlib import Path
import torch
import os
import json
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

from osprey.datasets.relation_category import RELATION_QUESTIONS
from osprey.datasets.psg import SG_QUESTIONS
from osprey.eval.psg.eval_psg import PSGEval

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

class PSGSGDetectionEval(PSGEval):
    ''' 
    Scene Graph Classification evaluation: Predict object labels and relationships with detected regions.
    '''

    def eval(self, root_path, pan_seg_path, ann_file, use_bbox_text=False, 
             temperature=0.2, top_p=1.0):

        sgs = []
        ann = json.load(open(ann_file))

        data = self.get_chunk(ann['data'])

        for idx, datum in enumerate(tqdm(data)):

            # Load Image
            img_path = os.path.join(root_path, datum['file_name'])
            image_id = str(datum['image_id'])
            height = datum['height']
            width = datum['width']

            # Gather regions
            boxes, segs, masks = self.create_mask_input(datum['regions'], height, width, sort_regions_by_largest=True, top_k=self.max_regions)

            # Add region prompts
            num_objects = len(segs)
            bbox_texts = [self.bbox_to_text(bbox[:4], height, width) for bbox in boxes] if use_bbox_text else None
            
            # Generate scene graph as list of object and relation labels. 
            object_outputs: list[str]; relation_outputs: list[str]
            object_outputs, relation_outputs = self.generate_scene_graph(img_path, torch.from_numpy(masks), bbox_texts=bbox_texts, 
                                                                         temperature=temperature, top_p=top_p) 

            # Map generations into classes.
            object_labels: list[int] = self.get_object_labels(object_outputs)
            prediction_triplets: list[tuple[int,int,str]] = self.get_relation_triplets(relation_outputs) # [subj, obj, relation_str]
            predicate_names: list[str] = [triplet[2] for triplet in prediction_triplets]
            predicate_labels: list[int] = self.get_predicate_labels(predicate_names)

            # Gather relation triplets
            prediction_names: list[tuple[int,int,str]] = []
            prediction_labels: list[tuple[int,int,int]] = []
            for triplet, label in zip(prediction_triplets, predicate_labels):
                name = self.predicate_classes[label]
                subj = triplet[0]
                obj = triplet[1]

                # region_id1, region_id2,  relation
                prediction_names.append([subj, obj, name])
                prediction_labels.append([subj, obj, label])

            if idx < 5:
                print("[Pred]")
                print('\n'.join(object_outputs))
                print('\n'.join(relation_outputs))
                print(prediction_labels)

                print("[GT]")
                print(datum['relations'])

            # GT annotations
            gt_boxes = np.array([a['bbox'] for a in datum['annotations']])
            gt_object_labels = [d['category_id'] for d in datum['annotations']]

            # Save results
            sgs.append({
                'image_id': image_id,
                'question_id': datum['question_id'],
                'width': width, 
                'height': height,
                'gt_labels': datum['relations'],
                'gt_boxes': gt_boxes,
                'gt_object_labels': gt_object_labels,
                'pred_boxes': boxes,
                'pred_raw': prediction_triplets,
                'pred_names': prediction_names,
                'pred_labels': prediction_labels,
                'pred_object_names': object_outputs,
                'pred_object_labels': object_labels,
            })

            if idx == 0:
                pprint(sgs[0])
                # break
            
        return sgs

class PSGFullSGDetectionEval(PSGEval):
    ''' 
    Scene Graph Detection evaluation: Predict object labels and relationships with detected regions.
    Regions with High IoUs with Reference regions are considered in evaluation as candidates.
    '''

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

            # Gather region masks
            boxes, segs, masks = self.create_mask_input(datum['regions'], height, width, sort_regions_by_largest=True, top_k=self.max_regions)

            # Add region prompts
            num_objects = len(segs)
            bbox_texts = [self.bbox_to_text(bbox[:4], height, width) for bbox in boxes] if use_bbox_text else None
            sg_outputs: str = self.generate_holistic_scene_graph(img_path, torch.from_numpy(masks), bbox_texts, temperature=temperature, top_p=top_p)
    
            if idx < 5:
                print("[Pred]")
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
                'pred_boxes': boxes,
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
        python -m osprey.eval.psg.eval_psg_sg_detection --model exp/relation_category_interaction_description/ \
            --max_regions 30 \
            --temperature 0.5 \
            --top_p 0.95 \
            --description_context \
            --region_mode segmentation \
            --output osprey/eval/results/psg/psg_sg_detection/relation_category_interaction_description/temp0.5_top0.95_max_regions_30.jsonl
 
        python -m osprey.eval.psg.eval_psg_sg_detection --model exp/sg_psg \
            --max_regions 30 \
            --temperature 0.5 \
            --top_p 0.95 \
            --full_sg \
            --output osprey/eval/results/psg/psg_sg_detection/psg_temp0.5_top0.95_max_regions_50.jsonl
        
        python -m osprey.eval.psg.eval_psg_sg_detection --model exp/multi_region_v3_cot/ \
            --max_regions 20 \
            --temperature 0.5 \
            --top_p 0.95 \
            --full_sg \
            --json osprey/eval/psg/psg_asv2_val_test_regions_sam_whole_seem_1.7.json \
            --output osprey/eval/results/psg/psg_sg_detection_sam_whole_seem_1.7/psg_temp0.5_top0.95_max_regions_20.jsonl

        python -m osprey.eval.psg.eval_psg_sg_detection --model exp/release/multi_region_v5_gqa_cot_bs16/ \
            --max_regions 30 \
            --temperature 0.5 \
            --top_p 0.95 \
            --description_context \
            --json osprey/eval/psg/psg_asv2_val_test_regions_sam_whole_seem_1.7.json \
            --output osprey/eval/results//psg/psg_sg_detection/sam_whole_seem_1.7/psg_temp0.5_top0.95_max_regions_30.jsonl
        
        python -m osprey.eval.psg.eval_psg_sg_detection --model exp/release/multi_region_v5_gqa_cot_bs16/ \
            --max_regions 30 \
            --temperature 0.5 \
            --top_p 0.95 \
            --full_sg \
            --json osprey/eval/psg/psg_asv2_val_test_regions_sam_whole_seem_1.7.json \
            --output osprey/eval/results//psg/psg_sg_detection/sam_whole_seem_1.7/psg_temp0.5_top0.95_max_regions_30_full_sg.jsonl
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
    eval_class = PSGFullSGDetectionEval if args.full_sg else PSGSGDetectionEval
    sg_eval = eval_class(args.model, 
                    max_regions=args.max_regions,        
                    bert_model=args.bert,
                    category=args.category,
                    debug=args.debug,
                    region_mode=args.region_mode,
                    use_object_description_context=args.description_context,
                    chunk_idx=args.chunk_idx,
                    num_chunks=args.num_chunks,
    )
    results = sg_eval.eval(args.img, args.pan_seg_img, args.json, args.use_bbox_text,
                                 args.temperature, args.top_p)

    os.makedirs(Path(args.output).parent, exist_ok=True)
    pd.DataFrame(results).to_json(args.output, orient='records', lines=True)
    print('Saved results to {}'.format(args.output))
 

