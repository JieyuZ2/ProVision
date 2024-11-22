import argparse
import torch
import os
import json
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from functools import partial
from transformers import AutoTokenizer, CLIPImageProcessor
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.mm_utils import tokenizer_image_token
from osprey.train.train import preprocess_multimodal
from osprey.train.train import DataArguments
from osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM

from osprey.datasets.llava import get_whole_image_mask
from osprey.eval.eval import LLAVAEval

import numpy as np
import argparse

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

class GQAEval(LLAVAEval):

    def eval(self, root_path, ann_file, temperature=0.2, top_p=1.0):
        data_all = pd.read_json(open(ann_file), lines=True).to_dict(orient='records')
        sgs = []

        data_all= self.get_chunk(data_all)

        for idx, data in enumerate(tqdm(data_all)):
            image_id = str(data['image_id'])
            img_path = os.path.join(root_path, image_id+'.jpg')
            image = Image.open(img_path).convert('RGB')
            w,h = image.size

            question = data['question']
            prompt = self.get_question_prompt(question)
            # masks = self.get_single_mask(h,w)
            # masks = torch.from_numpy(masks)
            
            init_inputs: dict = self.get_init_inputs(img_path,
                                        self.image_processor,
                                        prompt=prompt,
                                        masks=None
                                        )
            image = init_inputs['image']
            masks = None # init_inputs['masks'].cuda()

            conv = self.get_new_conv()
            qs = init_inputs['sources'][0][0]['value']
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            if idx == 0:
                print(qs)

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            outputs: str = self.get_outputs(image, input_ids, masks, self.stop_str, 
                                                temperature, top_p, max_new_tokens=128)
            answer = data['answer']
            if idx < 5:
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
        python -m osprey.eval.gqa.eval_gqa_no_region --model exp/multi_region_v3-a100 \
            --temperature 0.2 \
            --top_p 1.0 \
            --jsonl data/gqa/testdev_balanced_sam_seem_regions.jsonl \
            --output osprey/eval/results/gqa/testdev_balanced/llava_region/multi_region_v3-a100/temp0.2.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey CoT gqa evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--jsonl', help='path to gqa json file with regions', 
                        default='data/gqa/testdev_balanced_sam_seem_regions.jsonl')
    parser.add_argument('--img', help='path to gqa imgs', default='../images/gqa/images')
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.01)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)

    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(Path(args.output).parent, exist_ok=True)

    gqa_eval = GQAEval(args.model, chunk_idx=args.chunk_idx, num_chunks=args.num_chunks, debug=False)
    results = gqa_eval.eval(args.img, args.jsonl, args.temperature)
    acc = np.mean([d['pred_answer'] == d['answer'] for d in results])
    print('GQA acc: {}'.format(acc))
    pd.DataFrame(results).to_json(args.output, orient='records', lines=True)

