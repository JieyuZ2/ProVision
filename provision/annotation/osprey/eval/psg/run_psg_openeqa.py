import argparse
from pathlib import Path
import torch
import os
import json
import numpy as np
from tqdm import tqdm
from pprint import pprint
import pandas as pd
from typing import List, Dict, Tuple
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.mm_utils import tokenizer_image_token
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.train.train import DataArguments

from osprey.datasets.relation_category import RELATION_QUESTIONS, RELATION_DESCRIPTION_QUESTIONS
from osprey.datasets.psg import SG_QUESTIONS
from osprey.eval.psg.run_psg import SGRun, FullSGRun

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

DETAILED_DESCRIPTION = "Can you provide me with a detailed description of the region in the picture marked by {}?"

class OpenEQASGRun(SGRun):
    ''' 
    Generate relationships across other regions for each object
    '''
    def __init__(self, model_path, max_regions, region_mode, 
                 use_object_description_context=True, use_long_description=False, chunk_idx=0, num_chunks=1, debug=False):
        super().__init__(
            model_path, 
            max_regions=max_regions, 
            region_mode=region_mode, 
            use_object_description_context=use_object_description_context,
            chunk_idx=chunk_idx,
            num_chunks=num_chunks,
            debug=debug,
            )
        
        self.use_long_description = use_long_description
    
    def filter_processed_data(self, data, processed_image_ids, is_coco):
        print('Already processed {} images'.format(len(processed_image_ids)))
        new_data = [datum for datum in data if datum['image_id'] not in processed_image_ids]
        print('Number of data to process: {} -> {}'.format(len(data), len(new_data)))
        return new_data
    
    def eval(self, root_path, ann_file, output_file, is_coco, temperature=0.2, top_p=1.0, batch_size=10):

        results = []
        sgs = []
        ann = pd.read_json(ann_file, lines=True).to_dict(orient='records')

        data = self.get_chunk(ann)

        processed_image_ids = self.load_processed_ids(output_file)
        data = self.filter_processed_data(data, processed_image_ids, is_coco)

        for idx, datum in enumerate(tqdm(data)):
            image_id = datum['image_id']
            if image_id in processed_image_ids:
                continue
            height = datum['height']
            width = datum['width']

            # Gather regions
            regions = self.get_topk_largest_regions(datum['regions'], top_k=self.max_regions)
            if len(regions) == 0:
                continue
            boxes, segs = self.process_regions(regions) # np.array([a['bbox'] for a in datum['regions']]) # xyxy

            img_path = os.path.join(root_path, image_id)

            # Add region prompts
            num_objects = len(segs)
            if num_objects == 0:
                continue
            begin_string = self.get_region_string(n=num_objects)

            # Create masks and regions
            masks = self.create_masks(boxes, segs, height, width)
            masks = torch.from_numpy(masks)

            # Generate relations for each region.
            object_outputs = []
            region_outputs = []
            for id in tqdm(range(num_objects)):

                subj_region = 'region' + str(id+1)

                # Get Object Description
                if self.use_long_description:
                    description_question = DETAILED_DESCRIPTION.format(subj_region)
                else:
                    description_question = RELATION_DESCRIPTION_QUESTIONS[0].format(subj_region)
                prompt = begin_string + ' ' + description_question
                init_inputs: dict = self.get_init_inputs(img_path,
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
                outputs: str = self.get_outputs(image, input_ids, masks, self.stop_str, 
                                                temperature, top_p, max_new_tokens=128)
                object_outputs.append(outputs)

                # Get Relations Per Object
                relation_question = RELATION_QUESTIONS[0].format(subj_region)

                # Model was trained to generate description, then relations in conversation format.
                # We use generated object description as input to generate the rest of relations.
                if self.use_object_description_context:
                    conv = self.get_new_conv()
                    qs = init_inputs['sources'][0][0]['value']
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], outputs)
                    conv.append_message(conv.roles[0], relation_question)
                    conv.append_message(conv.roles[1], None)
                else:
                    prompt: str = begin_string + ' ' + relation_question
                    prompt = prompt.format(subj_region)
                    init_inputs: dict = self.get_init_inputs(img_path,
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
                outputs: str = self.get_outputs(image, input_ids, masks, self.stop_str, 
                                                temperature, top_p, max_new_tokens=128)
                outputs = f"{subj_region}: {outputs}"
                region_outputs.append(outputs)

                if idx == 0:
                    print(object_outputs[-1])
                    print(region_outputs[-1])

            # Parse prediction
            prediction_triplets: List[Tuple[int,int,str]] = self.get_relation_triplets(region_outputs)

            sgs.append({
                'image_id': image_id,
                'width': width, 
                'height': height,
                'pred_boxes': boxes,
                'pred_raw': prediction_triplets,
                'pred_object_names': object_outputs,
            })
            results += sgs

            if len(sgs) % batch_size == 0:
                sgs = self.flush_results(sgs, output_file)
        if sgs:
            sgs = self.flush_results(sgs, output_file)
    
        return sgs

class OpenEQAFullSgRun(FullSGRun):

    ''' Full Scene Graph Generator for OpenEQA'''

    def eval(self, root_path, ann_file, output_file, is_coco, temperature=0.2, top_p=1.0, batch_size=10):

        results = []
        sgs = []
        ann = pd.read_json(ann_file, lines=True).to_dict(orient='records')

        data = self.get_chunk(ann)

        processed_image_ids = self.load_processed_ids(output_file)

        for idx, datum in enumerate(tqdm(data)):
            image_id = datum['image_id']
            if image_id in processed_image_ids:
                continue
            height = datum['height']
            width = datum['width']

            # Gather regions
            regions = self.get_topk_largest_regions(datum['regions'], top_k=self.max_regions)
            if len(regions) == 0:
                continue
            boxes, segs = self.process_regions(regions) # np.array([a['bbox'] for a in datum['regions']]) # xyxy
            
            img_path = os.path.join(root_path, image_id)

            # Add region prompts
            num_objects = len(segs)
            begin_string = self.get_region_prompt(n=num_objects) 
            begin_string = self.get_region_string(n=num_objects)

            # Create masks and regions
            masks = self.create_masks(boxes, segs, height, width)
            masks = torch.from_numpy(masks)

            # Get Object Description
            prompt: str = begin_string + ' ' + SG_QUESTIONS[0]
            init_inputs: dict = self.get_init_inputs(img_path,
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
            if idx == 0:
                print(prompt)

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            outputs: str = self.get_outputs(image, input_ids, masks, self.stop_str, 
                                            temperature, top_p, max_new_tokens=1024)

            print("[Pred]")
            print(outputs)
            sgs.append({
                'image_id': image_id,
                'width': width, 
                'height': height,
                'pred_boxes': boxes,
                'pred_raw': outputs,
            })
            results += sgs

            if len(sgs) % batch_size == 0:
                sgs = self.flush_results(sgs, output_file)
        if sgs:
            sgs = self.flush_results(sgs, output_file)
    
        return sgs

if __name__ == "__main__":
    '''
        python -m osprey.eval.psg.run_psg_openeqa --model exp/multi_region_v5_gqa_cot_bs16/ \
            --json data/regions/openeqa_hm3d_sam_whole_regions.jsonl \
            --img ../data/open-eqa/data/frames/ \
            --temperature 0.5 \
            --top_p 0.95 \
            --max_regions 30 \
            --region_mode segmentation \
            --description_context \
            --output osprey/eval/results/regions/openeqa_hm3d/multi_region_v3_cot/temp0.5_top0.95_max_regions_30.jsonl
        
        # Generate Full SG Instead
        python -m osprey.eval.psg.run_psg_openeqa --model exp/multi_region_v3_cot \
            --json data/regions/openeqa_hm3d_sam_whole_regions.jsonl \
            --img ../data/open-eqa/data/frames/ \
            --full_sg \
            --temperature 0.5 \
            --top_p 0.95 \
            --max_regions 30 \
            --region_mode segmentation \
            --description_context \
            --output osprey/eval/results/regions/openeqa_hm3d/multi_region_v3_cot/temp0.5_top0.95_max_regions_30_full_sg.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey sg evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--json', help='path to region files', required=True)# default='data/sg/test_vg_sg_sam_hq.json')
    parser.add_argument('--img', help='path to imgs', required=True)
    parser.add_argument('--is_coco', action='store_true', help='Use COCO dataset for generation')
    parser.add_argument('--full_sg', action='store_true', help='full sg generate')

    # Region config
    parser.add_argument('--max_regions', type=int, default=99)
    parser.add_argument('--region_mode', type=str, default='segmentation', choices=['box', 'segmentation', 'box_segmentation'])
    parser.add_argument('--description_context', action='store_true', help='Use previously generated description as context to generate relations')
    parser.add_argument('--long_description', action='store_true', help='Use a long description instead.')

    # Gen config
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print('Saving results to {}'.format(args.output))
    eval_class = OpenEQAFullSgRun if args.full_sg else OpenEQASGRun
    is_coco = 'coco' in args.img
    print('Data is coco: {}'.format(is_coco))

    os.makedirs(Path(args.output).parent, exist_ok=True)
    run_region = eval_class(args.model, 
                    max_regions=args.max_regions,        
                    region_mode=args.region_mode,
                    use_object_description_context=True,
                    use_long_description=args.long_description,
                    debug=args.debug,
                    chunk_idx=args.chunk_idx,
                    num_chunks=args.num_chunks,
    )

    print('Saving results to {}'.format(args.output))
    results = run_region.eval(args.img, args.json, args.output, is_coco, args.temperature, args.top_p, batch_size=args.batch_size) 