import argparse
import torch
import os
from tqdm import tqdm
import pandas as pd
from typing import List, Dict
from pathlib import Path

from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.mm_utils import tokenizer_image_token
from osprey.train.train import DataArguments

from osprey.eval.eval import LLAVAEval

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

class VSREval(LLAVAEval):

    def eval(self, root_path, ann_file, temperature=0.2, top_p=1.0) -> List[Dict]:
        data_all = pd.read_json(open(ann_file), lines=True).to_dict(orient='records')
        sgs = []

        for idx, data in enumerate(tqdm(data_all)):
            image_id = data['image']
            img_path = os.path.join(root_path, image_id)
            image = Image.open(img_path).convert('RGB')
            w,h = image.size

            question = get_text(data)
            answer = ANSWERS[data['label']]

            prompt = self.get_question_prompt(question)

            # masks = self.get_single_mask(h,w)
            # masks = torch.from_numpy(masks)
            init_inputs = self.get_init_inputs(img_path,
                                        self.image_processor,
                                        prompt=prompt,
                                        masks=None,
                                        )

            image = init_inputs['image']
            masks = None
            # masks = init_inputs['masks'].cuda()

            conv = self.get_new_conv()
            qs = init_inputs['sources'][0][0]['value']
            if idx == 0:
                print(qs)

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            outputs: str = self.get_outputs(image, input_ids, masks, self.stop_str, 
                                                temperature, top_p, max_new_tokens=128, do_sample=False)

            if idx < 5:
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
        python osprey/eval/vsr/eval_vsr_no_region.py --model exp/multi_region_v3/ \
            --jsonl data/vsr/zeroshot_test_sam_seem_regions.jsonl \
            --temperature 0.01 \
            --top_p 1.0 \
            --output osprey/eval/results/vsr/llava_region/multi_region_v3/temp0.01.jsonl
        
    '''
    parser = argparse.ArgumentParser(description='osprey CoT gqa evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--jsonl', help='path to vsr jsonl file with regions', 
                        default='./data/vsr/zeroshot_test_sam_seem_regions.jsonl')
    parser.add_argument('--img', help='path to coco imgs', default='../images/coco/')
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)

    parser.add_argument('--log_wandb', action='store_true', help='Log results to wandb')
    parser.add_argument('--wandb_run_name', default=None, type=str, help='wandb run name to save. (default: wandb output file)')

    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    vsr_eval = VSREval(args.model, debug=False)
    results = vsr_eval.eval(args.img, args.jsonl, args.temperature)

    acc = np.mean([d['pred_answer'] == d['answer'] for d in results])
    print('VSR acc: {}'.format(acc))
    print('Saving result to.. {}'.format(args.output))

    run_key = os.path.basename(args.output)
    pd.DataFrame(results).to_json(args.output, orient='records', lines=True)
