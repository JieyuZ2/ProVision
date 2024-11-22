import argparse
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

from osprey.eval.annotate.generate_sg import SGRunner, OspreySG, load_osprey_model
from osprey.eval.utils import shard_data, is_hdf5_file, load_hdf5_group, save_results_to_hdf5, \
    load_files_from_dir, load_image_files_from_dir, load_jsonl
from osprey.eval.annotate.generate_regions import ImageDataset, LLaVAPretrainImageDataset

class ImageRegionDataset(ImageDataset):
    def __init__(self, region_input, image_dir,
                 num_shards=1, shard_index=0, identifier_key='id'):
        
        self.image_dir = image_dir
        self.identifier_key = identifier_key
        region_data, region_keys = self.load_region_data(region_input, num_shards, shard_index)
        self.region_data: dict | h5py.File = region_data # Dict of region data indexed by identifier
        self.region_keys: list = region_keys # List of identifiers
        logging.info(f"Loaded {len(self.region_keys)} regions from {region_input}")
    
    def load_region_data(self, region_input, num_shards=1, shard_index=0):
        ''' Load with sharding'''
        if is_hdf5_file(region_input):
            data = h5py.File(region_input, 'r')
            region_keys = sorted(list(data.keys()))
            region_keys = shard_data(region_keys, num_shards, shard_index)
        else:
            assert os.path.isdir(region_input), f"Region output must be a directory or hdf5 file"
            files = load_files_from_dir(region_input, exts=['.json'])
            logging.info(f"Loading regions from {len(files)} files")    
            data = [json.load(open(file)) for file in files]
            data = shard_data(data, num_shards, shard_index)
            region_keys = [r[self.identifier_key] for r in data]
            data = {r[self.identifier_key]: r for r in data}
        
        return data, region_keys
    
    def __len__(self):
        return len(self.region_keys)
    
    def __getitem__(self, idx) -> dict:

        # load regions
        region_key = self.region_keys[idx]
        item: dict = self.load_region_item(region_key)
        
        # load PL image
        image_path = os.path.join(self.image_dir, item["image_id"])
        image = Image.open(image_path).convert('RGB') # get PIL image for SG input
        item["image"] = image
        
        return item
    
    def load_region_item(self, region_key) -> dict:
        datum = self.region_data[region_key]
        if isinstance(datum, dict):
            region_item = datum
        else: # hdf5
            assert isinstance(datum, h5py.Group), f"Invalid data type {type(datum)}"
            region_item: dict = load_hdf5_group(datum)
        
        regions = []
        for region in region_item['regions']:
            if isinstance(region, str):
                region = ast.literal_eval(region)
            regions.append(region)
        region_item['regions'] = regions
                
        return region_item
    
    def filter_processed_ids(self, output):
        """ Filters out image data entries if their results already exist in the output directory """
        processed_ids: set = self._get_processed_ids(output)
        original_len = len(self.region_keys)
        self.region_keys = [key for key in self.region_keys if key not in processed_ids]
        if isinstance(self.region_data, dict):
            self.region_data = {key: self.region_data[key] for key in self.region_keys}
        logging.info(f"Already processed {len(processed_ids)} ids from '{output}', processing {len(self.region_keys)}/{original_len} ids")

if __name__ == '__main__':
    '''
    python -m osprey.eval.annotate.generate_sg_batch \
        --model exp/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_7b_stage3_v1_bs16/checkpoint-102315/ \
        --region_input region_results/llava-pretrain/sam_whole.hdf5 \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/images/llava/LLaVA-Pretrain/images \
        --output_file region_results/llava-pretrain/sam_whole_sg.hdf5
    
    python -m osprey.eval.annotate.generate_sg_batch \
        --model exp/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_7b_stage3_v1_bs16/checkpoint-102315/ \
        --region_input region_results/llava-pretrain/sam_whole.hdf5 \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/images/llava/LLaVA-Pretrain/images \
        --sg_mode detailed \
        --output_file region_results/llava-pretrain/sam_whole_sg_detailed.hdf5
    
    python -m osprey.eval.annotate.generate_sg_batch \
        --model exp/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_7b_stage3_v1_bs16/checkpoint-102315/ \
        --region_input region_results/vg_test/sam_whole_part_sam2.hdf5 \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/images/vg/VG_100K_all \
        --sg_mode detailed \
        --output_file region_results/vg_test/sam_whole_part_sam2_sg_detailed.hdf5
    
    '''
    parser = argparse.ArgumentParser(description='osprey sg generator', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--region_input', help='file or dir with precomputed regions', required=True) 
    parser.add_argument('--image_dir', help='path to images', required=True)
    parser.add_argument('--output_file', help='hdf5 file to save scene graph', required=True)
    parser.add_argument('--sg_mode', default='holistic', choices=['holistic', 'detailed'])

    # Model Region processing
    parser.add_argument('--max_regions', type=int, default=99)
    parser.add_argument('--region_mode', type=str, default='segmentation', choices=['box', 'segmentation', 'box_segmentation'])

    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards to split the data into")
    parser.add_argument("--shard_index", type=int, default=0, help="Index of the shard to process")
    parser.add_argument("--overwrite_output_file", action="store_true", help="Overwrite the output file if it exists")

    # Model Generation config
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.5)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=0.95)

    args = parser.parse_args()

    # Load data
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    dataset = ImageRegionDataset(region_input=args.region_input, image_dir=args.image_dir, num_shards=args.num_shards, shard_index=args.shard_index)
    dataset.filter_processed_ids(args.output_file)
    data: dict = dataset[0]

    # Load sg model
    model_path = args.model
    model, tokenizer = load_osprey_model(model_path, device='cuda')
    conv_mode = 'mistral_instruct' if 'mistral' in model_path else 'osprey_v1'
    osprey_sg = OspreySG(model, tokenizer, region_mode='segmentation', conv=conv_mode, max_regions=args.max_regions)

    # generate scene graph with regions
    sg_runner = SGRunner(osprey_sg)
    
    os.makedirs(Path(args.output_file).parent, exist_ok=True)
    n_to_print = 3
    for idx, data in enumerate(tqdm(dataset)):

        image: Image.Image = data['image']
        regions = data['regions']
        if len(regions) == 0:
            logging.info(f"No regions found for ID {data['id']}.")
        regions, sg_result, sg_text = sg_runner(image, regions, sg_mode=args.sg_mode)

        # Save result
        result = {
            'id': data['id'],
            'image_id': data['image_id'],
            'scene_graph': sg_result,
            'scene_graph_raw_text': sg_text,
            'regions': regions
        }
        result.update({k: v for k,v in data.items() if k.startswith('metadata')})
        if idx < n_to_print:
            logging.info(f"Result: {pformat(result)}")
        write_mode = 'w' if idx == 0 and args.overwrite_output_file else 'a'
        save_results_to_hdf5([result], args.output_file, group_key='id', write_mode=write_mode)



    