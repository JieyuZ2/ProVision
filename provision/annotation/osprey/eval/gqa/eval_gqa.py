import argparse
import torch
import os
import json
from tqdm import tqdm
import pandas as pd
from functools import partial
from transformers import AutoTokenizer, CLIPImageProcessor
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.mm_utils import tokenizer_image_token
from osprey.train.train import preprocess_multimodal
from osprey.train.train import DataArguments
from osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM

from osprey.datasets.cot_sg import GQACoTSGDataset, bboxToMask
from osprey.eval.eval import OspreyEval

from pycocotools import mask as maskUtils
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import argparse

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

SG_QUESTIONS = [
    'generate a scene graph to answer the question using a single word or phrase.',
]

QUESTIONS = [
    'Answer the question using a single word or phrase.',
]

class GQAEval(OspreyEval):
    def __init__(self, model_path, max_regions=36, no_cot=False, debug=False):
        
        super().__init__(model_path, no_cot, debug)
        self.max_regions = max_regions

    def eval(self, root_path, ann_file, temperature=0.2, top_p=1.0):
        data_all = pd.read_json(open(ann_file), lines=True).to_dict(orient='records')
        sgs = []
 
        for data in tqdm(data_all):
            image_id = str(data['image_id'])
            img_path = os.path.join(root_path,image_id+'.jpg')
            height = data['height']
            width = data['width']

            question = data['data']['question']
            question_id = data['data']['question_id']
            answer = data['data']['answer']
            object_ids = data['data']['object_ids']

            # process scene graph to create mask and regions
            sg = data['objects']
            sg_keys = list(sg.keys())
            object2id = {k: (idx+1) for idx, k in enumerate(sg_keys)}
            object_mapping: dict = data['object_mapping']

            region_mapping = {}
            masks = np.zeros((len(sg_keys), height, width))
            ref_string = ''
            for i, obj in enumerate(sg_keys): 
                
                region = sg[obj]
                region_id = object2id[obj]

                # create mask
                # if self.use_box:
                #     pred_mask = bboxToMask(region['bbox'],height,width)
                # else:
                pred_mask = self.annToMask(region['segmentation'],height,width)
                masks[i] = pred_mask

                # and reference regions
                if i < self.max_regions:
                    ref_string = ref_string +  f'region{region_id} <mask><pos>' + ','

                # example scene graph text for visualization purposes
                def textify_relation(relation: dict) -> str:
                    return f"{relation['name']} region{object2id[relation['object']]}"
                name: str = region['name']
                attributes: list = region.get('attributes', [])
                relations = [rel for rel in region.get('relations', [])
                             if isinstance(rel, dict) and rel.get('object', None) in object2id and rel['name'] not in self.ignored_relations]
                relations = [textify_relation(rel) for rel in relations]
                sg_dict = {'name': name, 'attributes': attributes, 'relations': relations}
                region_mapping[obj] = sg_dict
            
            # TODO: fix this ugly hack
            masks = masks[:self.max_regions]

            ref_string = ref_string[:-1]

            # Example output text
            output_sg_text = ''
            object_ids = sorted([object_mapping[obj] for obj in object_ids], key=lambda x: int(x))
            for obj in object_ids: 
                output_sg_text += f"region{object2id[obj]}: {region_mapping[obj]}\n"

            ref_prefix = 'Given <region> in the image,'
            if self.no_cot:
                q = QUESTIONS[0]
            else:
                q = SG_QUESTIONS[0]
            begin_string = ref_prefix.replace('<region>', ref_string) + ' ' + q
            prompt = begin_string + ' ' + question
            masks = torch.from_numpy(masks)

            init_inputs = self.get_init_inputs(img_path,
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
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=1024,
                    top_p=top_p,
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

            print("Question: ", question)
            print("[GT]")
            print(output_sg_text)
            print("answer: ", answer)
            print("[Pred]")
            print(outputs)

            pred_answer = outputs.split('Answer:')[-1].split(',')[0].rstrip('.').lower().strip()
            sgs.append({
                'image_id': image_id,
                'question_id': question_id, 
                'question': question,
                'answer': answer,
                'prompt': prompt,
                'output_sg_text': output_sg_text,
                'object_ids': object_ids,
                'output': outputs,
                'pred_answer': pred_answer,
                'width': width, 
                'height': height
            })

        return sgs


if __name__ == "__main__":
    '''
        python -m osprey.eval.gqa.eval_gqa --model exp/gqa_cot_sg_no_relation_stage3/ \
            --temperature 0.2 \
            --top_p 1.0 \
            --json data/gqa/val_balanced_aokvqa_cot_gqa_sam_hq.jsonl \
            --output osprey/eval/results/gqa/val_balanced_aokvqa/gqa_regions/gqa_cot_sg_no_relation_stage3_sam_hq-latest-temp0.2_top1.0.jsonl
        
        python -m osprey.eval.gqa.eval_gqa --model /net/nfs.cirrascale/prior/jamesp/Osprey/exp/stage3_gqa_cot_sg \
            --temperature 0.5 \
            --top_p 1.0 \
            --output /net/nfs.cirrascale/prior/jamesp/Osprey/osprey/eval/results/gqa/stage3_gqa_cot_sg_sam_hq-latest-temp0.5_top1.0.jsonl \
            --img /net/nfs.cirrascale/prior/jamesp/data/gqa/images \
            --json /net/nfs.cirrascale/prior/jamesp/Osprey/data/gqa/val_aokvqa_sg_sam_hq_output.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey CoT gqa evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--json', help='path to gqa json file with regions', 
                        default='data/gqa/val_balanced_aokvqa_cot_gqa_sam_hq.jsonl')
    parser.add_argument('--no_cot', action='store_true')
    parser.add_argument('--max_regions', type=int, default=99)
    parser.add_argument('--img', help='path to gqa imgs', default='/mmfs1/gscratch/raivn/jspark96/data/images/gqa/images')
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)
    args = parser.parse_args()

    gqa_eval = GQAEval(args.model, debug=False, no_cot=args.no_cot, max_regions=args.max_regions)
    results = gqa_eval.eval(args.img, args.json, args.temperature)
    acc = np.mean([d['pred_answer'] == d['answer'] for d in results])
    print('GQA acc: {}'.format(acc))
    pd.DataFrame(results).to_json(args.output, orient='records', lines=True)

