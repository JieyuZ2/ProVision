import argparse
import torch
import os
import json
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from functools import partial
from transformers import AutoTokenizer, CLIPImageProcessor
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.mm_utils import tokenizer_image_token
from osprey.train.train import preprocess_multimodal
from osprey.train.train import DataArguments
from osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM

from osprey.eval.aokvqa.eval_aokvqa import AOKVQAEval

from pycocotools import mask as maskUtils
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import argparse

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

LETTERS = ['A', 'B']

class SugarCrepeEval(AOKVQAEval):

    def __init__(self, model_path, chunk_idx=0, num_chunks=1, debug=False):
        super().__init__(model_path, chunk_idx, num_chunks, debug)
    
    def get_question_prompt(self, question, choices):
        choice_text: str = '\n'.join([f'{LETTERS[idx]}. {c}' for idx, c in enumerate(choices)])
        prompt = question + '\n' + choice_text + '\n' + self.question_str

        return prompt

    def eval(self, root_path, ann_file, temperature=0.2, top_p=1.0):

        with open(ann_file) as f:
            data_all: dict = json.load(f)
        data_all = list(data_all.items())
        sgs = []
        data_all = self.get_chunk(data_all)

        for idx, (question_id, data) in enumerate(tqdm(data_all)):
            img_path = os.path.join(root_path, data['filename'])
            image = Image.open(img_path).convert('RGB')
            w,h = image.size

            pos_cap = data['caption']
            neg_cap = data['negative_caption']
            for i in range(2):

                if i == 0:
                    choices = [pos_cap, neg_cap]
                    answer = 'A'
                else:
                    choices = [neg_cap, pos_cap]
                    answer = 'B'
                
                question = 'Which caption best describes the image?'
                prompt = self.get_question_prompt(question, choices)
                # masks = self.get_single_mask(h,w)
                # masks = torch.from_numpy(masks)

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

                qid = "{}_{}".format(question_id, i)
                pred_answer = outputs.split('Answer:')[-1].rstrip('.').strip()
                sgs.append({
                    'image_id': data['filename'],
                    'question_id': qid, 
                    'question': question,
                    'answer': answer,
                    'output': outputs,
                    'pred_answer': pred_answer,
                })

        return sgs
    
    def get_data(self, root_path, ann_file):

        with open(ann_file) as f:
            data_all: dict = json.load(f)
        data_all = list(data_all.items())
        sgs = []
        data_all = self.get_chunk(data_all)

        for idx, (question_id, data) in enumerate(tqdm(data_all)):
            img_path = os.path.join(root_path, data['filename'])
            image = Image.open(img_path).convert('RGB')
            w,h = image.size

            pos_cap = data['caption']
            neg_cap = data['negative_caption']
            for i in range(2):

                if i == 0:
                    choices = [pos_cap, neg_cap]
                    answer = 'A'
                else:
                    choices = [neg_cap, pos_cap]
                    answer = 'B'
                
                question = 'Which caption best describes the image?'
                choice_text: str = '\n'.join([f'{LETTERS[idx]}. {c}' for idx, c in enumerate(choices)])
                prompt = question + '\n' + choice_text + '\n' + self.question_str

                qid = "{}_{}".format(question_id, i)
                sgs.append({
                    'image_id': data['filename'],
                    'question_id': qid, 
                    'question': prompt,
                    'answer': answer,
                })

        return sgs

if __name__ == "__main__":
    '''
        python osprey/eval/crpe/eval_sugar_crepe.py --model exp/multi_region_v3_cot_bs16/ \
            --temperature 0.01 \
            --top_p 1.0 \
            --output osprey/eval/results/sugar_crepe/replace_rel/multi_region_v3_cot_bs16/temp0.01.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey CRPE evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--json', help='path to gqa json file with regions', 
                        default='../data/sugarcrepe/replace_rel.json')
    parser.add_argument('--img', help='path to coco imgs', default='../images/coco/val2017')
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    sugar_crepe_eval = SugarCrepeEval(args.model, debug=args.debug)
    results = sugar_crepe_eval.eval(args.img, args.json, args.temperature)

    acc = np.mean([d['pred_answer'] == d['answer'] for d in results])
    print('Sugar CREPE total acc: {}'.format(acc))
    print('Saving result to.. {}'.format(args.output))
    df = pd.DataFrame(results)
    df.to_json(args.output, orient='records', lines=True)

