import argparse
import torch
import os
import json
import random
from tqdm import tqdm
import pandas as pd
from typing import List, Dict
from pathlib import Path

from functools import partial
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.mm_utils import tokenizer_image_token
from osprey.train.train import DataArguments
from osprey.datasets.vcr import VCRDataset, CHOICES, WHY_QUESTIONS, MC_ANSWER, choices_to_text

from osprey.eval.eval import OspreyEval

import re
import numpy as np
from PIL import Image
import argparse

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

class VCREval(OspreyEval, VCRDataset):
    def __init__(self, model_path, 
                chunk_idx:int=0,
                num_chunks:int=1, debug=False):

        super().__init__(model_path, chunk_idx=chunk_idx, num_chunks=num_chunks, debug=debug)

    def eval(self, root_path, ann_file, qa_mode: str, use_bbox_text=False) -> List[Dict]:
        data_all = pd.read_json(open(ann_file), lines=True).to_dict(orient='records')
        sgs = []

        for idx, data in enumerate(tqdm(self.get_chunk(data_all))):

            # Load metadata
            metadata_fn_path = data['metadata_fn']
            img_fn = data['img_fn']
            img_path = os.path.join(root_path, img_fn)
            annotations = json.load(open(os.path.join(root_path, metadata_fn_path)))
            segms = annotations['segms']
            bboxes = np.array(annotations['boxes'])
            image = Image.open(img_path).convert('RGB')
            w,h = image.size

            # Get region propmt
            class_names = data['objects']
            num_objects = len(class_names)
            assert len(segms) == num_objects

            bbox_texts = [self.bbox_to_text(bbox[:4], h, w) for bbox in bboxes] if use_bbox_text else None
            begin_string = self.get_region_string(num_objects, bbox_texts=bbox_texts)
            # begin_string = self.get_region_prompt(num_objects)

            # Get region proposals
            masks = self.get_regions(bboxes, segms, w, h)
            masks = torch.from_numpy(masks)

            # Get QAR isetup
            q = self.replace_numbers_with_tags_tokens(data['question'], class_names)
            answer_choices = [self.replace_numbers_with_tags_tokens(a, class_names) for a in data['answer_choices']]
            answer_choices = choices_to_text(answer_choices)
            answer = CHOICES[data['answer_label']]

            rationale_choices = [self.replace_numbers_with_tags_tokens(a, class_names) for a in data['rationale_choices']]
            rationale_choices = choices_to_text(rationale_choices)

            # MC prompt
            prompt = begin_string + ' ' + q + '\n' + answer_choices + MC_ANSWER[0]

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
            if qa_mode == 'qa':
                conv.append_message(conv.roles[1], None)
                label = data['answer_label']
                choices = answer_choices
            elif qa_mode == 'qar':
                why_question = "What's the rationale for your decision?"
                conv.append_message(conv.roles[1], answer)  
                conv.append_message(conv.roles[0], why_question + '\n' + rationale_choices + MC_ANSWER[0])
                conv.append_message(conv.roles[1], None)
                label = data['rationale_label']
                choices = rationale_choices
            else:
                raise ValueError(f'Invalid qa_mode: {qa_mode}')
            prompt = conv.get_prompt()

            if idx == 0:
                print(prompt)

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            self.model.model.tokenizer = self.tokenizer

            with torch.inference_mode():

                self.model.orig_forward = self.model.forward
                self.model.forward = partial(self.model.orig_forward,
                                            img_metas=[None],
                                            masks=[masks.half()])
                output_ids = self.model.generate(
                    input_ids,
                    images=image.unsqueeze(0).half().cuda(),
                    do_sample=False,
                    max_new_tokens=1024,
                    use_cache=True,
                    num_beams=1,
                )
                self.model.forward = self.model.orig_forward

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (
                input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(
                    f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:],
                                                skip_special_tokens=True)[0]

            outputs = outputs.strip()
            if outputs.endswith(self.stop_str):
                outputs = outputs[:-len(self.stop_str)]
            outputs: str = outputs.strip()

            if idx < 5:
                print(prompt)
                print("[GT]")
                print(CHOICES[label])
                print("[Pred]")
                print(outputs)

            pred_answer = outputs.split('Answer:')[-1].rstrip('.').strip()
            if pred_answer not in CHOICES:
                pred_label = 0
            else:
                pred_label = CHOICES.index(pred_answer) 
            sgs.append({
                'image_id': img_fn,
                'question_id': data['annot_id'], 
                'question': q,
                'choices': choices,
                'mode': qa_mode,
                'answer': label,
                'pred_answer': pred_label,
            })
            
        return sgs

if __name__ == "__main__":
    '''
        python osprey/eval/vcr/eval_vcr.py --model exp/stage3_full_vcr/checkpoint-80976/ \
            --mode qa \
            --output osprey/eval/results/vcr/stage3_full_vcr-epoch1_qa.jsonl
        
        python osprey/eval/vcr/eval_vcr.py --model exp/gqa_cot_sg_no_relation_stage3/ \
            --mode qar \
            --output osprey/eval/results/vcr/stage3_full_vcr-epoch1_qar.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey CoT gqa evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--jsonl', help='path to vcr jsonl file with regions', 
                        default='./data/vcr/val.jsonl')
    parser.add_argument('--img', help='path to vcr imgs', default='../images/vcr/vcr1images')
    parser.add_argument('--output', help='path to save results to json file', required=True)
    parser.add_argument('--mode', type=str, help='qar mode', choices=['qa', 'qar'])
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument('--use_bbox_text', action='store_true')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--wandb_run_name', default=None, type=str, help='wandb run name to save. (default: wandb output file)')

    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    if 'bbox_text' in args.model:
        assert args.use_bbox_text, 'Model is trained with bbox text. Please use --use_bbox_text'
        
    vcr_eval = VCREval(
        args.model, 
        debug=args.debug,
        chunk_idx=args.chunk_idx,
        num_chunks=args.num_chunks,
    )
    results = vcr_eval.eval(args.img, args.jsonl, args.mode, use_bbox_text=args.use_bbox_text)

    acc = np.mean([d['pred_answer'] == d['answer'] for d in results])
    print('VCR {} acc: {}'.format(args.mode, acc))
    print('Saving result to.. {}'.format(args.output))

    run_key = os.path.basename(args.output)
    pd.DataFrame(results).to_json(args.output, orient='records', lines=True)
        

