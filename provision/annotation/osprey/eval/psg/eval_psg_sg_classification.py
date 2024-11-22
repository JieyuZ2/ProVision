import argparse
from pathlib import Path
import torch
import os
import json
import numpy as np
from tqdm import tqdm
from pprint import pprint
import logging
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

class PSGSGClassificationEval(PSGEval):
    ''' 
    Scene Graph Classification evaluation: Predict object labels and relationships given GT regions.
    Predict label and relationship for each region at a time.
    '''

    def eval(self, root_path, pan_seg_path, ann_file, use_bbox_text=False, 
             temperature=0.2, top_p=1.0):

        sgs = []
        ann = json.load(open(ann_file))

        data = self.get_chunk(ann['data'])

        for idx, datum in enumerate(tqdm(data)):
            image_id = str(datum['image_id'])
            height = datum['height']
            width = datum['width']

            # Gather regions
            boxes = np.array([a['bbox'] for a in datum['annotations']]) # xyxy
            pan_seg_image = os.path.join(pan_seg_path, datum['pan_seg_file_name'])
            segments_info = datum['segments_info']
            segs: np.ndarray = self.create_segmentations(pan_seg_image, segments_info)
            
            img_path = os.path.join(root_path, datum['file_name'])

            # Add region prompts
            num_objects = len(segs)
            bbox_texts = [self.bbox_to_text(bbox[:4], height, width) for bbox in boxes] if use_bbox_text else None
            begin_string = self.get_region_string(num_objects, bbox_texts=bbox_texts)
            # begin_string = self.get_region_prompt(n=num_objects)
            # begin_string = self.get_region_string(n=num_objects)

            # Create masks and regions
            masks = torch.from_numpy(segs)

            # Generate relations for each region.
            region_outputs = []
            for id in range(num_objects):
                prompt = begin_string + ' ' + RELATION_QUESTIONS[0]

                subj_region = 'region' + str(id+1)
                prompt = prompt.format(subj_region)
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

                # Generated relations per object
                outputs: str = self.get_outputs(image, input_ids, masks, self.stop_str, temperature, top_p, max_new_tokens=1024)
                outputs = f"{subj_region}: {outputs}"
                region_outputs.append(outputs)

                if idx == 0 and id == 0:
                    print(prompt)

            # Parse prediction
            # object_labels: list = self.get_object_labels(object_outputs)
            prediction_triplets: List[Tuple[int,int,str]] = self.get_relation_triplets(region_outputs)

            # Map predicate labels
            predicate_names = [triplet[2] for triplet in prediction_triplets]
            predicate_labels = self.get_predicate_labels(predicate_names)

            # final relation names
            prediction_names: List[Tuple[int,int,str]] = []
            prediction_labels: List[Tuple[int,int,int]] = []
            for triplet, label in zip(prediction_triplets, predicate_labels):
                name = self.predicate_classes[label]
                prediction_names.append([triplet[0], triplet[1], name])
                prediction_labels.append([triplet[0], triplet[1], label])

            if idx < 5:
                print("[Pred]")
                print('\n'.join(region_outputs))
                print(prediction_labels)

                print("[GT]")
                print(datum['relations'])

            gt_object_labels = [d['category_id'] for d in datum['annotations']]
            sgs.append({
                'image_id': image_id,
                'question_id': datum['question_id'],
                'width': width, 
                'height': height,
                'gt_labels': datum['relations'],
                'gt_boxes': boxes,
                'gt_object_labels': gt_object_labels,
                'pred_boxes': boxes,
                'pred_raw': prediction_triplets,
                'pred_names': prediction_names,
                'pred_labels': prediction_labels,
            })

            if idx == 0:
                pprint(sgs[0])
            
        return sgs

class PSGFullSGClassificationEval(PSGEval):
    ''' 
    Scene Graph Classification evaluation: Predict object labels and relationships given GT regions.
    Predict labels and relationships for all regions in a single inference pass.
    '''

    def eval(self, root_path, pan_seg_path, ann_file, use_bbox_text=False, 
             temperature=0.2, top_p=1.0):

        sgs = []
        ann = json.load(open(ann_file))

        data = self.get_chunk(ann['data'])

        for idx, datum in enumerate(tqdm(data)):
            image_id = str(datum['image_id'])
            height = datum['height']
            width = datum['width']

            # Gather regions
            boxes = np.array([a['bbox'] for a in datum['annotations']]) # xyxy
            pan_seg_image = os.path.join(pan_seg_path, datum['pan_seg_file_name'])
            segments_info = datum['segments_info']
            segs: np.ndarray = self.create_segmentations(pan_seg_image, segments_info)
            
            img_path = os.path.join(root_path, datum['file_name'])

            # Add region prompts
            num_objects = len(segs)
            bbox_texts = [self.bbox_to_text(bbox[:4], height, width) for bbox in boxes] if use_bbox_text else None
            begin_string = self.get_region_string(num_objects, bbox_texts=bbox_texts)
            # begin_string = self.get_region_prompt(n=num_objects) 
            # begin_string = self.get_region_string(n=num_objects)

            # Create masks and regions
            masks = torch.from_numpy(segs)

            # Generate entire scene graph.
            prompt: str = begin_string + ' ' + SG_QUESTIONS[0]
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

            if idx == 0:
                print(prompt)

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()    
            outputs: str = self.get_outputs(image, input_ids, masks, self.stop_str, 
                                            temperature, top_p, max_new_tokens=1024)

            # object_outputs, relation_outputs = outputs.split('Relations:')
            # object_outputs = object_outputs.split('Objects:')[1].strip()

            # Parse prediction
            # object_labels: list = self.get_object_labels(object_outputs)
            # prediction_triplets: List[Tuple[int,int,str]] = self.get_relation_triplets(region_outputs)

            # # Map predicate labels
            # predicate_names = [triplet[2] for triplet in prediction_triplets]
            # predicate_labels = self.get_predicate_labels(predicate_names)

            # # final relation names
            # prediction_names: List[Tuple[int,int,str]] = []
            # prediction_labels: List[Tuple[int,int,int]] = []
            # for triplet, label in zip(prediction_triplets, predicate_labels):
            #     name = self.predicate_classes[label]
            #     prediction_names.append([triplet[0], triplet[1], name])
            #     prediction_labels.append([triplet[0], triplet[1], label])


            gt_object_labels = [d['category_id'] for d in datum['annotations']]
            if idx < 5:
                print("[Pred]")
                print(outputs)
                print()

                print("[GT]")
                print("Objects:")
                for l, object_label in enumerate(gt_object_labels):
                    print(f"region{l+1}: {self.object_classes[object_label]}")
                print()
                print("Relations:")
                for relation in datum['relations']:
                    subj, obj, rel_class = relation
                    print(f"[region{subj+1}, region{obj+1}, {self.predicate_classes[rel_class]}]")

            sgs.append({
                'image_id': image_id,
                'question_id': datum['question_id'],
                'width': width, 
                'height': height,
                'gt_labels': datum['relations'],
                'gt_object_labels': gt_object_labels,
                'gt_boxes': boxes,
                'pred_boxes': boxes,
                'pred_raw': outputs,

                # 'pred_names': prediction_names,
                # 'pred_labels': prediction_labels,
            })

            if idx == 0:
                pprint(sgs[0])
            
        return sgs

if __name__ == "__main__":
    '''
        python -m osprey.eval.psg.eval_psg_sg_classification --model exp/relation_coco_sam_seem_box_segm_verified_0409/checkpoint-17000/ \
            --temperature 0.5 \
            --top_p 0.95 \
            --output osprey/eval/results/relation/psg_sg_det/relation_coco_sam_seem_box_segm_verified_0409-gt_objects_temp0.5_top0.95.jsonl
        
        python -m osprey.eval.psg.eval_psg_sg_classification --model exp/multi_region_v3 \
            --temperature 0.5 --top_p 0.95 --full_sg \
            --output osprey/eval/results/relation/psg_sg_det//multi_region_v3/temp0.5_top0.95_full_sg.jsonl

        python -m osprey.eval.psg.eval_psg_sg_classification --model exp/osprey_multi_region_v5_gqa_cot_bbox_text-osprey_stage_2-mistral-7b-instruct-v0.2_bs16/checkpoint-55423/ \
            --temperature 0.5 --top_p 0.95 --full_sg \
            --use_bbox_text \
            --output osprey/eval/results/relation/psg_sg_det/osprey_multi_region_v5_gqa_cot_bbox_text-osprey_stage_2-mistral-7b-instruct-v0.2_bs16/checkpoint-55423/temp0.5_top0.95_full_sg.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey sg evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--bert', help='path to bert model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--category', help='path to PSG Categories', default='osprey/eval/psg/psg_category.json')
    parser.add_argument('--json', help='path to gqa json file with regions', default='osprey/eval/psg/psg_asv2_val_test.json')# default='data/sg/test_vg_sg_sam_hq.json')
    parser.add_argument('--img', help='path to gqa imgs', default='../images/coco')
    parser.add_argument('--pan_seg_img', help='path to gqa imgs', default='../data/coco/panoptic_annotations')

    parser.add_argument('--region_mode', type=str, default='segmentation', choices=['box', 'segmentation', 'box_segmentation'])
    parser.add_argument('--full_sg', action='store_true', help='Generate full scene graph end to end')
    parser.add_argument('--use_bbox_text', action='store_true', help='Generate full scene graph end to end')

    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.5)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=0.95)
    parser.add_argument('--output', help='path to save results to json file', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print('Saving results to {}'.format(args.output))
    eval_class = PSGFullSGClassificationEval if args.full_sg else PSGSGClassificationEval
    relation_eval = eval_class(args.model, 
                    bert_model=args.bert,
                    category=args.category,
                    debug=args.debug,
                    region_mode=args.region_mode,
                    chunk_idx=args.chunk_idx,
                    num_chunks=args.num_chunks,
    )
    results = relation_eval.eval(args.img, args.pan_seg_img, args.json, args.use_bbox_text, args.temperature, args.top_p)
    os.makedirs(Path(args.output).parent, exist_ok=True)
    pd.DataFrame(results).to_json(args.output, orient='records', lines=True)
    print('Saved results to {}'.format(args.output))

