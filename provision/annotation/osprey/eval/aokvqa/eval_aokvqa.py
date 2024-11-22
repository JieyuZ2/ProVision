import argparse
import torch
import os
import json
from tqdm import tqdm
import pandas as pd
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

WANDB_KEY = 'eval/aokvqa_mc_acc'
LETTERS = ['A', 'B', 'C', 'D']

class AOKVQAEval(LLAVAEval):

    def __init__(self, model_path, chunk_idx=0, num_chunks=1, debug=False):
        super().__init__(model_path, chunk_idx, num_chunks, debug)

        # "What is the man by the bags awaiting?\nA. train\nB. delivery\nC. cab\nD. skateboarder\nAnswer with the option's letter from the given choices directly."
        self.question_str =  "Answer with the option's letter from the given choices directly."
    
    def get_question_prompt(self, question, choices):
        choice_text: str = '\n'.join([f'{LETTERS[idx]}. {c}' for idx, c in enumerate(choices)])
        prompt = self.region_str + question + '\n' + choice_text + '\n' + self.question_str

        return prompt

    def eval(self, root_path, ann_file, temperature=0.2, top_p=1.0):
        with open(ann_file, 'r') as f:
            data_all = json.load(f)

        sgs = []
        data_all = self.get_chunk(data_all)

        for data in tqdm(data_all):
            image_id = str(data['image_id']).zfill(12)
            img_path = os.path.join(root_path, image_id+'.jpg')
            image = Image.open(img_path).convert('RGB')
            w,h = image.size

            question = data['question']
            choices = data['choices']
            prompt = self.get_question_prompt(question, choices)
            masks = self.get_single_mask(h,w)
            masks = torch.from_numpy(masks)
            
            answer = LETTERS[data['correct_choice_idx']]

            init_inputs = self.get_init_inputs(img_path,
                                        self.image_processor,
                                        prompt=prompt,
                                        masks=masks
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
                                                temperature, top_p, max_new_tokens=128)

            print("Question: ", question)
            print("[GT]")
            print(question)
            print(choices)
            print(answer)
            print("[Pred]")
            print(outputs)

            pred_answer = outputs.split('Answer:')[-1].rstrip('.').strip()
            sgs.append({
                'image_id': image_id,
                'question_id': data['question_id'], 
                'question': question,
                'answer': answer,
                'output': outputs,
                'pred_answer': pred_answer,
            })

        return sgs

if __name__ == "__main__":
    '''
        python osprey/eval/aokvqa/aokvqa_eval.py --model exp/multi_region_v3_cot_bs16/ \
            --temperature 0.01 \
            --top_p 1.0 \
            --output osprey/eval/results/aokvqa/val/multi_region_v3_cot_bs16/temp0.01.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey CoT gqa evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--json', help='path to gqa json file with regions', 
                        default='../data/aokvqa/aokvqa_v1p0_val.json')
    parser.add_argument('--img', help='path to coco imgs', default='../images/coco/val2017/')
    parser.add_argument('--no_cot', action='store_true')
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)

    parser.add_argument('--wandb_run_name', default=None, type=str, help='wandb run name to save to if not None')

    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    aokvqa_eval = AOKVQAEval(args.model, debug=False)
    results = aokvqa_eval.eval(args.img, args.json, args.temperature)

    acc = np.mean([d['pred_answer'] == d['answer'] for d in results])
    print('AOKVQA acc: {}'.format(acc))
    print('Saving result to.. {}'.format(args.output))
    pd.DataFrame(results).to_json(args.output, orient='records', lines=True)

