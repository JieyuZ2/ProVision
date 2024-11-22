import argparse
import torch
import os
import math
import json
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.mm_utils import tokenizer_image_token, load_image_from_base64
from osprey.train.train import DataArguments

from osprey.eval.aokvqa.eval_aokvqa import AOKVQAEval

import numpy as np
import argparse

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

LETTERS = ['A', 'B', 'C', 'D']

def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

class MMBenchEval(AOKVQAEval):

    def eval(self, ann_file, temperature=0.2, top_p=1.0):
        data_all = pd.read_csv(ann_file, delimiter='\t').to_dict(orient='records')
        data_all = self.get_chunk(data_all)

        sgs = []
        for idx, data in enumerate(tqdm(data_all)):
            image = data['image']
            image = load_image_from_base64(image)
            w,h = image.size

            question = data['question']
            # Add Hint
            hint = data['hint']
            if not is_none(hint):
                question = hint + '\n' + question
            choices = [data[k] for k in ['A','B','C','D'] if not isinstance(data[k], float)]
            prompt = self.get_question_prompt(question, choices)
            answer = data['answer']

            init_inputs = self.get_init_inputs(image,
                                        self.image_processor,
                                        prompt=prompt,
                                        masks=None,
                                        # masks=masks
                                        )

            image = init_inputs['image']
            masks = None

            conv = self.get_new_conv()
            qs = init_inputs['sources'][0][0]['value']
            if idx == 0:
                print(qs)

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            outputs: str = self.get_outputs(image, input_ids, masks, self.stop_str, 
                                                temperature, top_p, max_new_tokens=128)

            if idx < 5:
                print("Question: ", question)
                print("[GT]")
                print(answer)
                print("[Pred]")
                print(outputs)

            pred_answer = outputs.split('Answer:')[-1].rstrip('.').strip()
            sgs.append({
                'image_id': data['index'],
                'question_id': data['index'], 
                'question': question,
                'answer': answer,
                'output': outputs,
                'category': data['l2-category'],
                'pred_answer': pred_answer,
            })

        return sgs

if __name__ == "__main__":
    '''
        python osprey/eval/bench/eval_mmbench.py --model exp/multi_region_v3_cot_bs16/ \
            --temperature 0.01 \
            --top_p 1.0 \
            --output osprey/eval/results/mmbench/multi_region_v3_cot_bs16/temp0.01.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey Bench evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--data', help='path to mmbench file', 
                        default='/net/nfs.cirrascale/mosaic/jamesp/data/mmbench/MMBench_DEV_EN_legacy.tsv')
    parser.add_argument('--no_cot', action='store_true')
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--wandb_run_name', default=None, type=str, help='wandb run name to save to if not None')

    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    bench_eval = MMBenchEval(args.model, debug=args.debug)
    results = bench_eval.eval(args.data, args.temperature)

    print('Saving result to.. {}'.format(args.output))
    df = pd.DataFrame(results)
    for v in df['category'].value_counts().keys():
        df_v = df[df['category'] == v]
        print(v, (df_v['pred_answer'] == df_v['answer']).mean())

    acc = np.mean([d['pred_answer'] == d['answer'] for d in results])
    print('MMBench total acc: {}'.format(acc))
    df.to_json(args.output, orient='records', lines=True)

