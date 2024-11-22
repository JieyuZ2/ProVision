import argparse
from copy import deepcopy
import json
from pathlib import Path
from PIL import Image
import logging
from pprint import pformat

import pandas as pd
from tqdm import tqdm
from functools import partial
import os
import ast
import numpy as np
import cv2
import h5py
from collections import defaultdict

from osprey.eval.utils import load_hdf5_file, save_results_to_hdf5
from osprey.eval.draw_utils import visualize_masks


sg_format = """
{
    'obj_id': {
        'name': 'A description of the object',
        'bbox': [x1, y1, x2, y2],
        'rel': {
            'rel_name': [integer list of other obj_ids],
        }
    },
}
"""

def get_response(task, gpt_model='gpt-4o-2024-08-06') -> str:
    id, image_id, scene_graph, image_path, regions, caption = task
    input_sg: str = str(json.loads(json.dumps(scene_graph)))
    images: list[Image.Image] = load_images(image_path, regions)
    im = images[0]

    for _ in range(2):
        result = openai_api.call_chatgpt_vision(
            model=gpt_model,
            sys_prompt=sys_prompt,
            usr_prompt=f"Caption:{caption}\n\nImage Size (width, height): {im.size}\nGenerated Scene Graph:\n{input_sg}",
            image_input=images,
            response_format=SGEdit,
            temperature=0.2,
            max_tokens=4096,
        )
        if result is None:
            return None
        response, usage = result
        edit_sg: dict = parse_edit_output(response)
        try:
            updated_sg: dict = apply_edit(scene_graph, edit_sg)
            total_tokens = usage.total_tokens
            completion_tokens = usage.completion_tokens
            return {
                'image_id': image_id,
                'id': id,
                'edit_sg': edit_sg,
                'updated_sg': updated_sg,
                'total_tokens': total_tokens,
                'completion_tokens': completion_tokens
            }
        except Exception as e:
            logging.error(f"Error in applying edits: {e}")
            logging.error(f"Edit: {edit_sg}")
            logging.info('Trying again...')
    
    # return None if failed
    return None

if __name__ == '__main__':
    '''

    # VG
    
    # PSG
    python -m osprey.eval.join_gpt_sg \
        --region_result region_results/psg_test/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_stage_3-mistral-7b-instruct-v0.2_bs16_epoch2/sam_whole_part_sam2_sg_detailed_temp0.5_top0.95_1_0.hdf5 \
        --gpt_result region_results/psg_test/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_stage_3-mistral-7b-instruct-v0.2_bs16_epoch2/sam_whole_part_sam2_sg_detailed_temp0.5_top0.95_1_0_gpt_sg.jsonl \
        --output_file region_results/psg_test/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_stage_3-mistral-7b-instruct-v0.2_bs16_epoch2/sam_whole_part_sam2_sg_detailed_temp0.5_top0.95_1_0_gpt_sg.hdf5
    
    '''
    parser = argparse.ArgumentParser(description='osprey sg generator', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--region_result', help='hdf5 with initial region results with sg', required=True)
    parser.add_argument('--gpt_result', help='jsonl file with generated scene graphs', required=True) 
    parser.add_argument('--output_file', help='hdf5 file to save joined results', required=True)
    args = parser.parse_args()

    assert args.output_file != args.gpt_result, "Output file cannot be the same as input file"
    assert args.output_file != args.region_result, "Output file cannot be the same as input file"

    # Load data
    sam_sg: dict = load_hdf5_file(args.region_result)
    for k, v in sam_sg.items():
        v['regions'] = [ast.literal_eval(region) for region in v['regions']]
    gpt_sg = pd.read_json(args.gpt_result, lines=True, dtype=False)

    breakpoint()
    