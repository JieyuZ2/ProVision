import argparse
import ast
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
from PIL import Image

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm import tqdm
import logging

from osprey.eval.annotate.generate_regions import ImageDataset, load_dataset_class

from data_generation.openai_utils import OpenaiAPI, MultiProcessCaller

# set logging format
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

@dataclass
class CaptioningResult:
    id: str
    image_id: str
    text: list[dict]
    objects: list[str]

def single_task(datum: dict, gpt_model: str) -> dict:
    json_format = """ 
    {
        'text': only the dense text paragraph of the image.
        'objects': list of all objects that should be included from your description.
    }
"""
    image = datum['image']
    im = Image.fromarray(image)
    response = openai_api.call_chatgpt_vision(
        model=gpt_model, # 'gpt-4o-2024-08-06',
        sys_prompt="""
        Provide a dense description of the image detailing every significant objects and their relations or interactions it has with other objects.
        Try to segment the different events and parts in the image and provide a detailed description of each event.
        Lastly, give a list of objects that I should include definitely include so that I can ground and provide bounding boxes for them. 
        Do not mention anything from the prompt in your response.
        """,
        usr_prompt=f"Provide your response for this image in JSON. {json_format}",
        image_input=im,
        response_format='json',
        temperature=0.5,
        max_tokens=2048,
        # response_format=Description
    )
    if response is None:
        return None
    gpt_result, usage = response    

    for _ in range(3):
        try:
            gpt_result = ast.literal_eval(gpt_result)
            caption: str = gpt_result['text']
            objects: list[str] = gpt_result['objects']
            result = {
                'id': datum["id"],
                'image_id': datum["image_id"],
                'caption': caption,
                'objects': objects,
                'completion_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens,
            }
            for k in datum.keys():
                if k.startswith("metadata"):
                    result[k] = datum[k]
            
            return result
        except Exception as e:
            logging.error(f"Error in parsing GPT response: {e}")
            logging.error(f"Response: {gpt_result}")
    return None
    

if __name__ == '__main__':
    '''
    python -m osprey.eval.annotate.run_gpt_captioning \
        --dataset_name llava_pretrain \
        --image_data /net/nfs.cirrascale/mosaic/jamesp/images/llava/LLaVA-Pretrain/blip_laion_cc_sbu_558k_meta.json \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/images/llava/LLaVA-Pretrain/images \
        --output gpt_results/llava-pretrain/captions_gpt-4o-2024-05-13.jsonl

    python -m osprey.eval.annotate.run_gpt_captioning \
        --dataset_name llava_pretrain \
        --image_data /net/nfs.cirrascale/mosaic/jamesp/images/llava/LLaVA-Pretrain/blip_laion_cc_sbu_558k_meta.json \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/images/llava/LLaVA-Pretrain/images \
        --gpt_model gpt-4o-2024-08-06 \
        --output gpt_results/llava-pretrain/captions_gpt-4o-2024-08-06.jsonl
         
    # v3_det
    python -m osprey.eval.annotate.run_gpt_captioning \
        --dataset_name v3_det \
        --image_data /net/nfs.cirrascale/mosaic/jamesp/data/V3Det/annotations/v3det_2023_v1_train.json \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/data/V3Det/images \
        --output gpt_results/v3_det/captions_v3det_2023_v1_train.jsonl
    
    # vg-test
    python -m osprey.eval.annotate.run_gpt_captioning \
        --dataset_name vg \
        --image_data /net/nfs.cirrascale/mosaic/jamesp/data/vg/test_image_data.json \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/images/vg/VG_100K_all \
        --gpt_model gpt-4o-2024-08-06 \
        --output gpt_results/vg_test/captions_gpt-4o-2024-08-06.jsonl
    
    # psg-test
    python -m osprey.eval.annotate.run_gpt_captioning \
        --dataset_name psg \
        --image_data /net/nfs.cirrascale/mosaic/jamesp/Osprey/osprey/eval/psg/psg_asv2_val_test.json \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/images/coco/ \
        --gpt_model gpt-4o-2024-08-06 \
        --output gpt_results/psg_test/captions_gpt-4o-2024-08-06.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey sg generator', formatter_class=argparse.RawTextHelpFormatter)
    # Model config
    parser.add_argument('--gpt_model', help='gpt model', default='gpt-4o-2024-05-13', choices=['gpt-4o', 'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06'])
    parser.add_argument('--dataset_name', help='image json file', required=True,) 
    parser.add_argument('--image_data', help='image json file', required=True) 
    parser.add_argument('--image_dir', help='path to images', required=True)
    parser.add_argument('--output', help='jsonl file to save results', default=None)

    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards to split the data into")
    parser.add_argument("--shard_index", type=int, default=0, help="Index of the shard to process")
    parser.add_argument("--overwrite_output_file", action="store_true", help="Overwrite the output file if it exists")

    # Dataloader
    parser.add_argument('--num_workers', help='num_workers', default=32, type=int)
    parser.add_argument('--batch_size', help='batch size', default=1000, type=int)

    args = parser.parse_args()

    # Load data
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    dataset_class = load_dataset_class(args.dataset_name)
    dataset: ImageDataset = dataset_class(image_dir=args.image_dir, image_file=args.image_data,
                                        num_shards=args.num_shards, shard_index=args.shard_index,)
    if not args.overwrite_output_file:
        dataset.filter_processed_ids(args.output)
    # data: list = list(dataset.get_data().values())

    # Load GPT Runner
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    openai_api = OpenaiAPI()
    _single_task = partial(single_task, gpt_model=args.gpt_model)
    MultiProcessCaller.batch_process_save(
        data=dataset,
        openai_call=_single_task,
        num_processes=args.num_workers,
        output_file=args.output,
        sort_key=dataset.identifier_key,
        batch_size=args.batch_size, 
        write_mode='w' if args.overwrite_output_file else 'a',
    )
    
    
    



    
    

    
        





    
    