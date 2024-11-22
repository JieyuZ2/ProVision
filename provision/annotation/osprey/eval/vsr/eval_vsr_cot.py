import argparse
import torch
import os
import json
from tqdm import tqdm
import pandas as pd
from typing import List, Dict
from pathlib import Path

from functools import partial
from transformers import AutoTokenizer, CLIPImageProcessor
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.mm_utils import tokenizer_image_token
from osprey.train.train import preprocess_multimodal
from osprey.train.train import DataArguments
from osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM

from osprey.eval.eval import OspreyEval

from osprey.datasets.cot_sg import GQACoTSGDataset, SG_QUESTIONS, QUESTIONS
from pycocotools import mask as maskUtils
import numpy as np
from PIL import Image
import argparse

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

ANSWERS = ['no', 'yes']

def get_text(datum: dict):
    if datum['caption'].split().count('is') == 1:
        text = f"Is the {datum['subj']} {datum['relation']} the {datum['obj']}?"
    else:
        relation = datum['relation']
        if relation == 'has as a part':
            text = f"Is the {datum['subj']} part of the {datum['obj']}?"
        else:
            if relation[-1] == 's':
                relation = relation[:-1]
            text = f"Does the {datum['subj']} {relation} the {datum['obj']}?"
    return text

class VSREval(OspreyEval, GQACoTSGDataset):
    def __init__(self, model_path, region_mode, max_regions=36, debug=False, no_cot=False):

        self.region_mode = region_mode
        self.max_regions = max_regions
        self.no_cot = no_cot
        super().__init__(model_path, debug=debug)

    def eval(self, root_path, ann_file, temperature=0.2, top_p=1.0) -> List[Dict]:
        data_all = pd.read_json(open(ann_file), lines=True).to_dict(orient='records')
        sgs = []

        for idx, data in enumerate(tqdm(data_all)):
            image_id = data['image']
            img_path = os.path.join(root_path, image_id)
            image = Image.open(img_path).convert('RGB')
            w,h = image.size
            regions = data['regions'][:self.max_regions]

            question = get_text(data)

            region_string = self.get_region_string(len(regions))
            if self.no_cot:
                q = QUESTIONS[0]
            else:
                q = SG_QUESTIONS[0]

            answer = ANSWERS[data['label']]

            begin_string = region_string + ' ' + q
            prompt = begin_string + ' ' + question

            boxes, segs = self.process_regions(regions)
            masks = self.create_masks(boxes, segs, h, w)
            masks = torch.from_numpy(masks)
            if idx == 0:
                print(prompt)

            init_inputs = self.get_init_inputs(img_path,
                                        self.image_processor,
                                        prompt=prompt,
                                        masks=masks,
                                        )

            image = init_inputs['image']
            masks = init_inputs['masks'].cuda()

            conv = self.get_new_conv()
            qs = init_inputs['sources'][0][0]['value']

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            outputs: str = self.get_outputs(image, input_ids, masks, self.stop_str, 
                                                temperature, top_p, max_new_tokens=128, do_sample=False)

            print("Question: ", question)
            print("[GT]")
            print(question)
            print(answer)
            print("[Pred]")
            print(outputs)

            pred_answer = outputs.split('Answer:')[-1].rstrip('.').lower().strip()
            if pred_answer.lower() not in ANSWERS:
                pred_answer = 'yes' if pred_answer.lower() in question.lower() else 'no'
            sgs.append({
                'image_id': image_id,
                'question_id': data['index'], 
                'question': question,
                'answer': answer,
                'output': outputs,
                'pred_answer': pred_answer,
            })

        return sgs
    
    def eval_results(results: List[Dict]):
        ''' Evaluate result '''
        acc = np.mean([d['pred_answer'] == d['answer'] for d in results])

        return {
            'accuracy': acc
        }


def get_pred_answers(df):
    pred_answers = []
    for pred_answer, question in zip(df['pred_answer'], df['question']):
        pred_answer = pred_answer.lower()
        if pred_answer not in ['yes', 'no']:
            pred_answer = 'yes' if pred_answer.lower() in question.lower() else 'no'
        pred_answers.append(pred_answer)
    return pred_answers


if __name__ == "__main__":
    '''
        python osprey/eval/vsr/eval_vsr_cot.py --model exp/gqa_cot_sg_grounded/ \
            --jsonl ./data/vsr/zeroshot_test_sam_seem_regions.jsonl \
            --temperature 0.2 \
            --top_p 1.0 \
            --output osprey/eval/results/vsr/sam_seem_regions/gqa_cot_sg_grounded/temp0.2_top1.0.jsonl
        
        python osprey/eval/vsr/eval_vsr_cot.py --model exp/gqa_cot_sg_grounded/ \
            --jsonl ./data/vsr/zeroshot_test_coco_objects_sam_hq.jsonl \
            --temperature 0.2 \
            --top_p 1.0 \
            --output osprey/eval/results/vsr/coco_regions/gqa_cot_sg_grounded/temp0.2_top1.0.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey CoT gqa evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--jsonl', help='path to vsr jsonl file with regions', 
                        default='./data/vsr/zeroshot_test_sam_regions.jsonl')
    parser.add_argument('--img', help='path to coco imgs', default='../images/coco/')
    parser.add_argument('--region_mode', type=str, default='segmentation', choices=['box', 'segmentation', 'box_segmentation'])
    parser.add_argument('--no_cot', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)

    parser.add_argument('--log_wandb', action='store_true', help='Log results to wandb')
    parser.add_argument('--wandb_run_name', default=None, type=str, help='wandb run name to save. (default: wandb output file)')

    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    vsr_eval = VSREval(args.model, region_mode=args.region_mode, debug=args.debug, no_cot=args.no_cot)
    results = vsr_eval.eval(args.img, args.jsonl, args.temperature)

    acc = np.mean([d['pred_answer'] == d['answer'] for d in results])
    print('VSR acc: {}'.format(acc))
    print('Saving result to.. {}'.format(args.output))

    run_key = os.path.basename(args.output)
    pd.DataFrame(results).to_json(args.output, orient='records', lines=True)

