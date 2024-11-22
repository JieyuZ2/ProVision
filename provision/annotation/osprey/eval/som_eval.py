
import json
import os
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Tuple
import math
import torch

from tqdm import tqdm
from functools import partial
from pathlib import Path

import argparse
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.mm_utils import tokenizer_image_token
from osprey.conversation import conv_templates, SeparatorStyle

from osprey.eval.psg.eval_psg import PSGEval

import sys
sys.path.insert(0, os.path.join(Path(__file__).parent.parent.parent, 'som'))
from som.som_inference import build_som_model, inference as inference_som

DETAILED_DESCRIPTION = "Can you provide me with a detailed description of the region in the picture marked by {}?"
RELATION_QUESTION = "Generate list of relationships for: {}."
RELATION_DESCRIPTION_QUESTION = "Generate a description for: {}."

def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RegionProposal:
    def __init__(self, model_path: str, device='cuda'):
        self.model_path = model_path
        self.device = get_default_device()
        self.model = self.load_model()
    
    def generate(self, image: Image.Image) -> List[Dict[str, Any]]:
        raise NotImplementedError

class SomOspreyEval(PSGEval):
    def __init__(
            self, 
            model_path: str, 
            max_regions: int, 
            region_mode: str, 
            use_object_description_context: bool=False,
    ):
        super().__init__(model_path, 
                         max_regions=max_regions, 
                         region_mode=region_mode, 
                         use_object_description_context=use_object_description_context,
        )
    
    def inference(self, image, temperature=0.2, top_p=1.0,):
        regions: list = self.generate_regions(image)
        return self.inference_from_regions(image, regions, temperature=temperature, top_p=top_p)
    
    def inference_from_regions(self, image, regions, temperature=0.2, top_p=1.0):
        """Inference using the given image and regions.

        Args:
            image (type): The input image for the inference.
            regions (type): The regions of interest for the inference.

        Returns:
            type: The result of the inference.
        """

        im: Image.Image = self.load_image(image)
        width, height = im.size
        
        # Gather regions
        regions = self.get_topk_largest_regions(regions, top_k=self.max_regions)
        boxes, segs = self.process_regions(regions) 

        # Create mask input
        masks = self.create_masks(boxes, segs, height, width)
        masks = torch.from_numpy(masks)

        # Create region prompt
        num_objects = len(segs)
        if num_objects == 0:
            print('No regions for generated for image {}'.format(image))
            return None
        begin_string = self.get_region_prompt(n=num_objects)

        # Generate object description and relations for each region.
        region_outputs = []


        # cache key value
        # self.model.model.tokenizer = self.tokenizer
        # init_inputs: dict = self.get_init_inputs(im,
        #                                     self.image_processor,
        #                                     masks=masks,
        #                                     prompt=begin_string,
        #                                     )
        # image = init_inputs['image']
        # masks = init_inputs['masks'].cuda()

        # conv = self.get_new_conv()
        # qs = init_inputs['sources'][0][0]['value'] # Initial prompt wiht image tokens
        # conv.append_message(conv.roles[0], qs)
        # prompt = conv.get_prompt()
        # cache_input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        # cache_input_ids = cache_input_ids[:,:-1]

        # with torch.inference_mode():
        #     self.model.orig_forward = self.model.forward
        #     self.model.forward = partial(
        #         self.model.orig_forward,
        #         img_metas=[None],
        #         masks=[masks.half()]
        #     )
        #     outputs = self.model(
        #         cache_input_ids,
        #         images=image.unsqueeze(0).half().cuda(),
        #         use_cache=True
        #     )

        #     self.model.forward = self.model.orig_forward

        #     past_key_values = outputs.past_key_values

        for id in tqdm(range(num_objects)):

            region = regions[id]
            subj_region = 'region' + str(id+1)

            ''' Get Object Description '''
            description_question = RELATION_DESCRIPTION_QUESTION.format(subj_region)
            object_prompt: str = begin_string + ' ' + description_question

            # init_inputs = self.get_init_inputs(im, self.image_processor, masks=masks, prompt=object_prompt) 
            # image = init_inputs['image']
            # masks = init_inputs['masks'].cuda()

            # conv = self.get_new_conv()
            # qs = init_inputs['sources'][0][0]['value'] # Initial prompt wiht image tokens
            # conv.append_message(conv.roles[0], qs)
            # conv.append_message(conv.roles[1], None)
            # prompt = conv.get_prompt()

            # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            # with torch.no_grad():
            #     output_ids = self.model.generate(
            #         input_ids[:,cache_input_ids.size(1):],
            #         images=image.unsqueeze(0).half().cuda(),
            #         img_metas=[None],
            #         masks=[masks.half()],
            #         past_key_values=past_key_values,
            #         do_sample=True,
            #         temperature=temperature,
            #         max_new_tokens=128,
            #         top_p=top_p,
            #         use_cache=True,
            #         num_beams=1,
            #     )
            object_output: str = self.generate(im, masks, object_prompt, temperature, top_p, max_new_tokens=128, return_dict_in_generate=True)

            ''' Get Relations'''
            relation_question = RELATION_QUESTION.format(subj_region)
            if self.use_object_description_context: # Use generated object description as context when generating relations.
                relation_object_prompt = [
                    object_prompt,              # user input
                    object_output,              # mode output
                    relation_question           # user input
                ]
                relation_output = self.generate(im, masks, relation_object_prompt, temperature, top_p, max_new_tokens=128)
            else: # Generate relations from scratch without object description.
                relation_prompt: str = begin_string + ' ' + relation_question
                relation_output = self.generate(im, masks, relation_prompt, temperature, top_p, max_new_tokens=128)
            relation_output = f"{subj_region}: {relation_output}"

            if True or id == 0:
                print("[Pred]")
                print(object_output)
                print(relation_output)

            region_outputs.append({
                'region_id': id,
                'region': region,
                'pred_object_name': object_output,
                'pred_relations': relation_output
            })

        return region_outputs
        
    
if __name__ == "__main__":
    '''
        python -m osprey.eval.som_eval --model  exp/multi_region_v5_gqa_cot_bs16/ \
            --img $DATA/PVSG_dataset/ego4d/frames/ec2e69c1-fd07-48ec-adff-0b2cf3ab25b6/0000.png \
            --max_regions 50 \
            --temperature 0.5 \
            --top_p 0.95 \
            --description_context \
            --region_mode segmentation \
            --output som_eval_test.json
    '''
    parser = argparse.ArgumentParser(description='osprey sg evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--img', help='path to single image', required=True)

    # Som Config
    parser.add_argument('--som_model_name', type=str, default='sam', choices=['sam', 'semantic-sam'])
    parser.add_argument('--som_device', type=str, default='cuda:1')
    parser.add_argument('--sam_mode', choices=['whole', 'default', 'part'], help='mode to run', default='whole')
    parser.add_argument('--slider', type=float, default=2.0) # sem sam params

    # Region config
    parser.add_argument('--max_regions', type=int, default=30)
    parser.add_argument('--region_mode', type=str, default='segmentation', choices=['box', 'segmentation', 'box_segmentation'])
    parser.add_argument('--description_context', action='store_true', help='Use previously generated description as context to generate relations')
    parser.add_argument('--full_sg', action='store_true', help='Generate full scene graph end to end')

    # Gen config
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)

    args = parser.parse_args()

    print('Saving results to {}'.format(args.output))


    # one for som model I suppose...?
    relation_eval = SomOspreyEval(
                    args.model, 
                    max_regions=args.max_regions,        
                    region_mode=args.region_mode,
                    use_object_description_context=args.description_context,
    )
    som_model = build_som_model(args.som_model_name)
    _, regions = inference_som(args.img, som_model, sam_mode=args.sam_mode, slider=args.slider, output_mode='coco_rle')
    result = relation_eval.inference_from_regions(args.img, regions, args.temperature, args.top_p)
    breakpoint()
    with open(args.output, 'w') as f:
        json.dump(result, f)

    

    