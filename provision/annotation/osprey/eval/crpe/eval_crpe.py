import argparse
import torch
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.mm_utils import tokenizer_image_token

from osprey.eval.eval import LLAVAEval

import numpy as np
from PIL import Image
import argparse


class CRPEEval(LLAVAEval):

    def __init__(self, model_path, chunk_idx=0, num_chunks=1, debug=False):
        super().__init__(model_path, chunk_idx, num_chunks, debug)

    def get_question_prompt(self, question):
        prompt = question

        return prompt

    def eval(self, coco_path, ann_file, temperature=0.2, top_p=1.0):
        data_all = pd.read_json(ann_file, lines=True).to_dict(orient='records')

        root_path = Path(ann_file).parent
        sgs = []
        data_all = self.get_chunk(data_all)

        for idx, data in enumerate(tqdm(data_all)):
            image = data['image']
            if 'coco' in image:
                img_path = os.path.join(coco_path, image)
            else:
                img_path = os.path.join(root_path, image)
            image = Image.open(img_path).convert('RGB')
            w,h = image.size

            question = data['text']
            prompt = self.get_question_prompt(question)
            # masks = self.get_single_mask(h,w)
            # masks = torch.from_numpy(masks)
            
            answer = data['correct_option']

            init_inputs = self.get_init_inputs(img_path,
                                        self.image_processor,
                                        prompt=prompt,
                                        masks=None
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
                                                temperature, top_p, max_new_tokens=128, do_sample=False)

            if idx < 5:
                print("Question: ", question)
                print("[GT]")
                print(question)
                print(answer)
                print("[Pred]")
                print(outputs)

            pred_answer = outputs.split('Answer:')[-1].rstrip('.').strip()
            sgs.append({
                'image_id': data['image'],
                'question_id': data['question_id'], 
                'question': question,
                'answer': answer,
                'output': outputs,
                'category': data['category'],
                'pred_answer': pred_answer,
            })

        return sgs

if __name__ == "__main__":
    '''
        python osprey/eval/crpe/eval_crpe.py --model exp/multi_region_v3_cot_bs16/ \
            --output osprey/eval/results/crpe/relation/multi_region_v3_cot_bs16/temp0.01.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey CRPE evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--json', help='path to gqa json file with regions', 
                        default='/net/nfs.cirrascale/mosaic/jamesp/data/CRPE/crpe_relation.jsonl')
    parser.add_argument('--img', help='path to coco imgs', default='../images')
    parser.add_argument('--no_cot', action='store_true')
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    crpe_eval = CRPEEval(args.model, chunk_idx=args.chunk_idx, num_chunks=args.num_chunks, debug=args.debug)
    results = crpe_eval.eval(args.img, args.json, args.temperature)

    acc = np.mean([d['pred_answer'] == d['answer'] for d in results])
    print('CRPE total acc: {}'.format(acc))
    print('Saving result to.. {}'.format(args.output))
    df = pd.DataFrame(results)
    for v in df['category'].value_counts().keys():
        df_v = df[df['category'] == v]
        print(v, (df_v['pred_answer'] == df_v['answer']).mean())
    df.to_json(args.output, orient='records', lines=True)

