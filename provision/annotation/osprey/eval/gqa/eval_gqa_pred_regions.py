import argparse
import torch
import os
import json
import random
from tqdm import tqdm
from typing import Literal
import pandas as pd
from pathlib import Path

from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.mm_utils import tokenizer_image_token
from osprey.train.train import DataArguments

from osprey.eval.eval import OspreyEval

from osprey.datasets.cot_sg import GQACoTSGDataset, SG_QUESTIONS, QUESTIONS
import numpy as np
from PIL import Image
import argparse

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

WANDB_KEY = 'eval/gqa_testdev_acc'

class GQAEval(OspreyEval, GQACoTSGDataset):
    def __init__(self, model_path, chunk_idx=0, num_chunks=1, debug=False, max_regions=150, region_mode: Literal['box', 'segmentation', 'box_segmentation']='segmentation',
                 no_cot=False, sort_regions=True, sort_by='area'):

        self.region_mode = region_mode
        self.no_cot = no_cot
        self.max_regions = max_regions
        self.sort_regions = sort_regions
        self.sort_by = sort_by
        super().__init__(model_path, chunk_idx, num_chunks, debug)

    def eval(self, root_path, ann_file, temperature=0.2, top_p=1.0):
        data_all = pd.read_json(open(ann_file), lines=True).to_dict(orient='records')
        sgs = []

        data_all= self.get_chunk(data_all)

        for idx, data in enumerate(tqdm(data_all)):
            image_id = str(data['image_id'])
            img_path = os.path.join(root_path, image_id+'.jpg')
            image = Image.open(img_path).convert('RGB')
            w,h = image.size
            regions = data['regions']

            # priorizie including large regions
            if self.sort_regions:
                regions = sorted([d for d in regions], key=lambda x: -x[self.sort_by])
            regions = regions[:self.max_regions]

            question = data['question']

            # Input Prompt
            region_string = self.get_region_string(len(regions))
            if self.no_cot:
                q = QUESTIONS[0]
            else:
                q = SG_QUESTIONS[0]

            begin_string = region_string + ' ' + q
            prompt = begin_string + ' ' + question

            boxes, segs = self.process_regions(regions)
            masks = self.create_masks(boxes, segs, h, w)
            masks = torch.from_numpy(masks)
            
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
                                                temperature, top_p, max_new_tokens=128)
            answer = data['answer']
            print("Question: ", question)
            print("[GT]")
            print(answer)
            print("[Pred]")
            print(outputs)

            pred_answer = self.parse_answer(outputs)
            sgs.append({
                'image_id': image_id,
                'question_id': data['question_id'], 
                'question': question,
                'prompt': prompt,
                'answer': answer,
                'output': outputs,
                'pred_answer': pred_answer,
            })

        return sgs
    
    def parse_answer(self, outputs: str) -> str:
        return outputs.split('Answer:')[-1].split(',')[0].rstrip('.').lower().strip()


if __name__ == "__main__":
    '''
        python osprey/eval/gqa/eval_gqa_pred_regions.py --model exp/gqa_cot_sg_no_relation_stage3/ \
            --jsonl ./data/gqa/testdev_balanced_sam_regions.jsonl \
            --temperature 0.2 \
            --top_p 1.0 \
            --output osprey/eval/results/gqa/testdev/gqa_cot_sg_no_relation_stage3-latest-temp0.2_top1.0.jsonl
        
        python osprey/eval/gqa/eval_gqa_pred_regions.py --model exp/gqa_cot_sg_no_relation_stage3/ \
            --jsonl ./data/gqa/val_balanced_aokvqa_sam_pred_regions.jsonl \
            --temperature 0.2 \
            --top_p 1.0 \
            --output osprey/eval/results/gqa/val_balanced_aokvqa/sam_regions/gqa_cot_sg_no_relation_stage3-sam_hq-latest-temp0.2_top1.0.jsonl
        
        python osprey/eval/gqa/eval_gqa_pred_regions.py --model exp/gqa_cot_sg_no_relation_stage3/ \
            --jsonl ../unified-sg/results/gqa/regions/val_balanced_aokvqa_gt_regions.jsonl \
            --temperature 0.2 \
            --top_p 1.0 \
            --no_sort_regions \
            --output osprey/eval/results/gqa/val_balanced_aokvqa/gqa_regions/gqa_cot_sg_no_relation_stage3-sam_hq-latest-temp0.2_top1.0_v2.jsonl

        python osprey/eval/gqa/eval_gqa_pred_regions.py --model exp/gqa_cot_sg_grounded/ \
            --jsonl data/gqa/testdev_balanced_sam_seem_regions.jsonl \
            --temperature 0.2 \
            --top_p 1.0 \
            --no_sort_regions \
            --output osprey/eval/results/gqa/testdev_balanced/sam_seem_regions/gqa_cot_sg_grounded/temp0.2.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey CoT gqa evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--jsonl', help='path to vsr jsonl file with regions', 
                        default='./data/gqa/testdev_balanced_sam_seem_regions.jsonl')
    parser.add_argument('--img', help='path to coco imgs', default='../images/gqa/images/')
    parser.add_argument('--max_regions', type=int, default=99)
    parser.add_argument('--no_cot', action='store_true')
    parser.add_argument('--region_mode', type=str, default='segmentation', choices=['box', 'segmentation', 'box_segmentation'])
    parser.add_argument('--no_sort_regions', action='store_true')
    parser.add_argument('--sort_by', choices=['area', 'stability_score'], default='area')
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)

    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)


    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    gqa_eval = GQAEval(args.model, debug=False, no_cot=args.no_cot, 
                       max_regions=args.max_regions,
                       sort_regions=not args.no_sort_regions,
                       sort_by=args.sort_by,
                       region_mode=args.region_mode,
                       chunk_idx=args.chunk_idx,
                       num_chunks=args.num_chunks,
                       )
    results = gqa_eval.eval(args.img, args.jsonl, args.temperature)

    acc = np.mean([d['pred_answer'] == d['answer'] for d in results])
    print('GQA testdevacc: {}'.format(acc))
    print('Saving result to.. {}'.format(args.output))
    pd.DataFrame(results).to_json(args.output, orient='records', lines=True)

