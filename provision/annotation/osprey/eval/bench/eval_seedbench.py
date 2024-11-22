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
from osprey.train.train import DataArguments
from osprey.mm_utils import tokenizer_image_token
from osprey.eval.aokvqa.eval_aokvqa import AOKVQAEval

import numpy as np
from PIL import Image
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

class SeedBenchEval(AOKVQAEval):

    def eval(self, root_path, ann_file, temperature=0.2, top_p=1.0):
        data_all = json.load(open(ann_file))['questions']
        # Get image only
        data_all = [d for d in data_all if d['data_type'] == 'image']
        data_all = self.get_chunk(data_all)

        sgs = []
        for idx, data in enumerate(tqdm(data_all)):
            image_id = data['data_id']
            img_path = os.path.join(root_path, image_id)
            image = Image.open(img_path).convert('RGB')
            w,h = image.size

            question = data['question']
            choices = [data['choice_'+k] for k in ['a','b','c','d'] if not isinstance(data['choice_'+k], float)]
            prompt = self.get_question_prompt(question, choices)
            # masks = self.get_single_mask(h,w)
            # masks = torch.from_numpy(masks)

            
            answer = data['answer']

            init_inputs = self.get_init_inputs(image,
                                        self.image_processor,
                                        prompt=prompt,
                                        masks=None # masks
                                        )

            image = init_inputs['image']
            masks = None # init_inputs['masks'].cuda()

            conv = self.get_new_conv()
            qs = init_inputs['sources'][0][0]['value']
            if idx == 0:
                print(qs)

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            outputs: str = self.get_outputs(image, input_ids, masks, self.stop_str, 
                                                max_new_tokens=128, do_sample=False)

            if idx < 5:
                print("Question: ", question)
                print("[GT]")
                print(question)
                print(answer)
                print("[Pred]")
                print(outputs)

            pred_answer = outputs.split('Answer:')[-1].rstrip('.').strip()
            sgs.append({
                'image_id': data['data_id'],
                'question_id': data['question_id'], 
                'question': question,
                'answer': answer,
                'output': outputs,
                'category': data['question_type_id'],
                'pred_answer': pred_answer,
            })

        return sgs

if __name__ == "__main__":
    '''
        python osprey/eval/bench/eval_seedbench.py --model exp/multi_region_v3_cot_bs16/ \
            --output osprey/eval/results/seedbench/multi_region_v3_cot_bs16/temp0.01.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey Bench evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--json', help='path to json mc file', 
                        default='/net/nfs.cirrascale/mosaic/jamesp/data/seedbench/SEED-Bench.json')
    parser.add_argument('--img', help='path to coco imgs', default='/net/nfs.cirrascale/mosaic/jamesp/data/seedbench/SEED-Bench-image')
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.01)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)

    parser.add_argument('--wandb_run_name', default=None, type=str, help='wandb run name to save to if not None')

    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    bench_eval = SeedBenchEval(args.model, chunk_idx=args.chunk_idx, num_chunks=args.num_chunks, debug=args.debug)
    results = bench_eval.eval(args.img, args.json, args.temperature)

    print('Saving result to.. {}'.format(args.output))
    df = pd.DataFrame(results)
    for v in df['category'].value_counts().keys():
        df_v = df[df['category'] == v]
        print(v, (df_v['pred_answer'] == df_v['answer']).mean())

    acc = np.mean([d['pred_answer'] == d['answer'] for d in results])
    print('SeedBench total acc: {}'.format(acc))
    df.to_json(args.output, orient='records', lines=True)

