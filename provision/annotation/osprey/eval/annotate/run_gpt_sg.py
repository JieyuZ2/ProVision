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

from osprey.eval.utils import load_hdf5_file
from osprey.eval.draw_utils import visualize_masks

from data_generation.openai_utils import OpenaiAPI, MultiProcessCaller

from pydantic import BaseModel
from typing import Optional

class Rel(BaseModel):
    rel: str
    obj_ids: list[int]
    
class Edit(BaseModel):
    obj_id: str
    name: str
    add: list[Rel] # (relation, other_object), e.g. ('on', 'table')
    remove: list[Rel]

class SGEdit(BaseModel):
    edit: list[Edit]

input_format = """
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

edit = """
{
    'obj_id': { 
        'name': accurate description    # description with correct object name and more specificity (e.g. first, lower-right, color, texture). Include all actions being performed by the object.
        'remove': [
            {'rel_name': [subset of obj_ids]},
            ...
        ], # subset of object_ids to be removed for the specified relation. the object_ids for this relation must be present in the input scene graph already. Please return empty list if no relations need to be added.
        'add': [
            {'rel_name': [subset of obj_ids]},
            ...
        ]     # list of relations to be added. Please comprehensively include all relationships. This SHOULD HAVE AT LEAST LENGTH 2. Please return empty list if no relations need to be added.
    },
}
"""

sys_prompt = f"""
You are given the original image and the same image with highlighted regions specified by id labels which share the same color with the referred region.
You are provided dense caption of the image that describes the scene as detailed as possible.
You are then provided generated object descriptions and their relationships in the scene graph that might contain hallucinated objects and relationships.
Your goal is to fix and improve this generated scene graph.
The objects in the scene graph are labeled with id labels in the image, and provided xyxy bounding box coordinates [x1,y1,x2,y2] referring to top left and bottom right of the box.
We want to verify the correctness of the objects and relationships in the scene graph.
The input scene graph has the format:
{input_format}

Your task: 
Generate ''just the modification'' to correct the objects and relationships in the scene graph.
Your response will be called by python dictionary `update` method, so make sure you generate just the necessary edits.
- Go through all the objects and determine if you can confidently say the object mentioned by noisy description is really present and visible in the image. 
    If you are unable to confidently verify the presence of an object or suspect it might be inaccurately represented, 
    please provide a revised description using a more accurate object name if possible, or alternatively, classify it under a broader, safer category. 
    These can be 'decoration', 'tool', 'foliage', 'furniture item', etc., to ensure clarity and avoid misidentification.
    Then, provide more speficity to your name so that viewer can disambiguate the object from other objects.
    Try to include the object's position, color, texture, and/or any other distinguishing feature.
    Remember to use the first original image to accurately identify the color and texture of the objects. 
- Then, make edits to the relationships by adding the prominent relationships current scene graph is missing,
    or/and removing erroneous relationships between objects in the scene. 
    For example, object 1 has relationship: {{'on': [2,3,4,5]}} and you think it should be {{'on': [2,3]}},
    then you should include object_ids [4,5] in the 'remove' list.
    Also, if object 2 has relationship: {{'contains': [2,6,7]}} and you think it should be {{'contains': [2,6,7,8]}},
    then you should include object_ids [8] in the 'add' list.
    If no relationships need to be added and/or removed, please return an empty list for 'add' and/or 'remove'.

Output should be a flattened JSON object wit no unnecessary spacing and format:
{edit}

Note:
- The object id labels are unique and will not be repeated.
- Use the bounding box coordinates, and highlighted regions in the image to precisely link the referred id to the object and locate them in the image.
- The size of the bounding box should inform you about the size of the object in the image to determine if the object name was correct.
- The second, third, forth, fifth images are provided to help you visualize the specified objects in the scene graph.
- Please make sure to provide a 'name' that has the correct description for every hallucinated objects, and use broader categories if the object is not clear.
- Remember that you can add and remove multiple relationship types for each object. IT IS CRUCIAL that you Fill all the relationships you think are missing and remove the ones that are incorrect.
- Use spatial markers like 'first', 'second', 'lower-left', 'top-right', 'middle', 'center', 'left', 'right', 'top', 'bottom', etc., to accurately describe the position of the object in the image.
- Fix any incorrect actions described in the object names, e.g., 'holding', 'eating', 'running', etc, and in relationship names.
- Make sure to include prominent and salient actions, events, and elements happening in the scene to your description and relationships.
- If no new relationships need to be added or removed, please return an empty list respectively. 
- Include at least two relationships to be added for each object in your response.
- In your 'add' relation, include the relation: "same as" if you think the id is referring to the same object as another id, e.g. {{"same as": [2]}}
"""

def format_scene_graph(scene_graph, regions: list) -> dict:
    ''' Format generated scene graph to more readable format '''
    objects = scene_graph['objects']
    relationships = scene_graph['relations']
    
    sg = {}
    for id, obj in enumerate(objects):
        region = regions[id]['bbox']
        xyxy = [region[0], region[1], region[0] + region[2], region[1] + region[3]]
        node = {
            'name': obj,
            'bbox': xyxy,
            'rel': defaultdict(list)
        }
        sg[id] = node
    for relationship in relationships:
        node_id, target_id, edge = relationship
        sg[node_id]['rel'][edge].append(target_id)
    
    # sort object_ids in relationships
    for id, node in sg.items():
        for rel, targets in node['rel'].items():
            sg[id]['rel'][rel] = sorted(list(set(targets)))
        
    return sg

def load_images(image_path, regions) -> list[Image.Image]:
    # Load mask inputs
    im = Image.open(image_path).convert('RGB')
    image_rgb = np.asarray(im)
    drawn_mask1 = visualize_masks(image_rgb, regions, draw_bbox=True, draw_mask=True,  white_padding=50, plot_image=False)
    drawn_mask1 = Image.fromarray(drawn_mask1)
    drawn_mask2 = visualize_masks(image_rgb, regions, draw_bbox=False, draw_mask=False, draw_polygon=True, white_padding=50, plot_image=False)
    drawn_mask2 = Image.fromarray(drawn_mask2)
    drawn_mask3 = visualize_masks(image_rgb, regions, draw_bbox=True, draw_mask=True,  white_padding=50, reverse_order=True, plot_image=False)
    drawn_mask3 = Image.fromarray(drawn_mask3)
    drawn_mask4 = visualize_masks(image_rgb, regions, draw_bbox=False, draw_mask=False, draw_polygon=True, white_padding=50, reverse_order=True, plot_image=False)
    drawn_mask4 = Image.fromarray(drawn_mask4)
    
    images = [im, drawn_mask1, drawn_mask2, drawn_mask3, drawn_mask4]
    return images

def parse_edit_output(response: SGEdit) -> dict:
    edits = response.edit
    edit_dict = {}
    for edit in edits:
        obj_id = edit.obj_id
        edit_dict[obj_id] = {
            'name': edit.name,
            'add': {rel.rel: rel.obj_ids for rel in edit.add},
            'remove': {rel.rel: rel.obj_ids for rel in edit.remove}
        }
    return edit_dict

def apply_edit(scene_graph: dict, edit_sg: dict) -> dict:
    ''' Apply edits to scene graph '''

    gen_sg = deepcopy(scene_graph)
    for obj_id, edit in edit_sg.items():
        obj_id = int(obj_id)

        # update object name if provided
        if edit['name']:
            gen_sg[obj_id]['name'] = edit['name']
        
        # add relations
        for rel, add_ids in edit['add'].items():
            gen_sg[obj_id]['rel'][rel] += add_ids
        
        # remove relations
        for rel, remove_ids in edit['remove'].items():
            og = gen_sg[obj_id]['rel'][rel]
            gen_sg[obj_id]['rel'][rel] = [id for id in gen_sg[obj_id]['rel'][rel] if id not in remove_ids]
            # print(og, gen_sg[obj_id]['rel'][rel], remove_ids)

    # remove empty relations
    for obj_id, node in gen_sg.items():
        for rel, targets in list(node['rel'].items()):
            if not targets:
                del node['rel'][rel]
            else:
                node['rel'][rel] = sorted(list(set(targets)))
    
    return gen_sg


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

    # Llava-Pretrain
        

    # VG
    python -m osprey.eval.annotate.run_gpt_sg \
        --input_file region_results/vg_test/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_stage_3-mistral-7b-instruct-v0.2_bs16_epoch2/sam_whole_part_sam2_sg_detailed_temp0.5_top0.95_1_0.hdf5 \
        --gpt_captions  gpt_results/vg_test/captions_gpt-4o-2024-08-06.jsonl \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/images/vg/VG_100K_all \
        --output_file region_results/vg_test/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_stage_3-mistral-7b-instruct-v0.2_bs16_epoch2/sam_whole_part_sam2_sg_detailed_temp0.5_top0.95_1_0_gpt_sg.jsonl 
    
    # PSG
    python -m osprey.eval.annotate.run_gpt_sg \
        --input_file region_results/psg_test/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_stage_3-mistral-7b-instruct-v0.2_bs16_epoch2/sam_whole_part_sam2_sg_detailed_temp0.5_top0.95_1_0.hdf5 \
        --gpt_captions  gpt_results/psg_test/captions_gpt-4o-2024-08-06.jsonl \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/images/coco/ \
        --output_file region_results/psg_test/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_stage_3-mistral-7b-instruct-v0.2_bs16_epoch2/sam_whole_part_sam2_sg_detailed_temp0.5_top0.95_1_0_gpt_sg.jsonl 
    
    '''
    parser = argparse.ArgumentParser(description='osprey sg generator', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_file', help='file with generated scene graphs', required=True) 
    parser.add_argument('--image_dir', help='path to images', required=True)
    parser.add_argument('--gpt_captions', help='path to gpt generated captions', required=True)
    parser.add_argument('--output_file', help='file to save gpt results', required=True)
    parser.add_argument('--gpt_model', help='gpt model', default='gpt-4o-2024-08-06')

    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards to split the data into")
    parser.add_argument("--shard_index", type=int, default=0, help="Index of the shard to process")
    parser.add_argument("--overwrite_output_file", action="store_true", help="Overwrite the output file if it exists")

    # Model Generation config
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)

    parser.add_argument('--num_workers', help='num_workers', default=32, type=int)
    parser.add_argument('--batch_size', help='batch size', default=1000, type=int)

    args = parser.parse_args()

    assert args.output_file != args.input_file, "Output file cannot be the same as input file"

    # Load data
    sam_sg: dict = load_hdf5_file(args.input_file)
    for k, v in sam_sg.items():
        v['regions'] = [ast.literal_eval(region) for region in v['regions']]
    
    # openai utils
    openai_api = OpenaiAPI()

    gpt_captions = pd.read_json(args.gpt_captions, lines=True, dtype=False)
    # gpt_captions['image_id'] = gpt_captions['image_id'].apply(lambda x: x.split('/')[-1].split('.')[0])
    gpt_captions: dict = gpt_captions.set_index('id')['caption'].T.to_dict()

    # Gather common image_ids
    image_ids = sorted(list(set(sam_sg.keys()).intersection(gpt_captions.keys())))
    print(f"Processing {len(image_ids)} images")

    tasks = []
    image_dir = Path(args.image_dir)
    for image_id in image_ids:

        # Load inputs
        item = sam_sg[image_id]
        id = item['id']
        sg_image_id = item['image_id']  # Use the original image_id from sg
        regions = item['regions']
        image_path = image_dir / sg_image_id
        scene_graph = ast.literal_eval(item['scene_graph'])
        scene_graph: dict = format_scene_graph(scene_graph, regions)
        caption = gpt_captions[image_id]

        assert len(regions) == len(scene_graph), f"Length mismatch: {len(regions)} != {len(scene_graph)}"
        
        task = (id, sg_image_id, scene_graph, image_path, regions, caption)
        tasks.append(task)
    
    _get_response = partial(get_response, gpt_model=args.gpt_model)
    _get_response(tasks[0])
    MultiProcessCaller.batch_process_save(
        data=tasks,
        openai_call=_get_response,
        num_processes=args.num_workers,
        output_file=args.output_file,
        sort_key='id',
        batch_size=args.batch_size, 
        write_mode='w' if args.overwrite_output_file else 'a',
    )
    