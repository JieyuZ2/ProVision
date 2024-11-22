import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from PIL import Image

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import logging
from transformers import pipeline

from osprey.eval.annotate.generate_regions import ImageDataset, load_dataset_class
from osprey.eval.utils import is_hdf5_file, save_results_to_hdf5

# set logging format
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

@dataclass
class DepthResult:
    id: str
    image_id: str
    regions: list[dict]
    
class DepthPipeline:
    def __init__(self, depth_model):
        self.pipeline = pipeline(task="depth-estimation", model=depth_model, device='cuda')
    
    def _generate_depth_results(self, datum, depth) -> dict:
        """ Returns results to save"""
        result = {
            'id': datum["id"],
            'image_id': datum["image_id"],
            'depth': depth
        }
        for k in datum.keys():
            if k.startswith("metadata"):
                result[k] = datum[k]
        
        return result
    
    def __call__(self, image_dataset: dict | list[dict] | ImageDataset, 
                 disable_tqdm: bool = False
        ) -> list[dict]:
        results = []

        if isinstance(image_dataset, dict):
            image_dataset = [image_dataset]

        for datum in tqdm(image_dataset, disable=disable_tqdm):
            image: np.ndarray = datum["image"]
            image = Image.fromarray(image)

            depth: Image.Image = self.pipeline(image)["depth"]
            result = self._generate_depth_results(datum, depth)
            results.append(result)

        return results
    
    def process_and_save_batches(self, data_loader: DataLoader, output, disable_tqdm: bool = False) -> list[dict]:
        for idx, batches in enumerate(tqdm(data_loader, disable=disable_tqdm)):
            results: list[dict] = self.__call__(batches, disable_tqdm=disable_tqdm)
            self.save_results(results, output)
            logging.info(f"Saved {len(results)} results to {output}")
    
    def save_results(self, results: list[dict], output: str):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        save_results_to_hdf5(results, output)

if __name__ == '__main__':
    '''
    python -m osprey.eval.annotate.run_depth \
        --dataset_name llava_pretrain \
        --image_data /net/nfs.cirrascale/mosaic/jamesp/images/llava/LLaVA-Pretrain/blip_laion_cc_sbu_558k_meta.json \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/images/llava/LLaVA-Pretrain/images \
        --num_shards 16 \
        --shard_index 0 \
        --output annotate_results/depth_results/llava-pretrain/blip_laion_cc_sbu_558k_16_0.hdf5

    # v3_det
    python -m osprey.eval.annotate.run_depth \
        --dataset_name v3_det \
        --image_data /net/nfs.cirrascale/mosaic/jamesp/data/V3Det/annotations/v3det_2023_v1_train.json \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/data/V3Det/images \
        --output annotate_results/depth_results/v3_det/v3det_2023_v1_train.hdf5
       
    # vg_test
    python -m osprey.eval.annotate.run_depth \
        --dataset_name vg \
        --image_data /net/nfs.cirrascale/mosaic/jamesp/data/vg/test_image_data.json \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/images/vg/VG_100K_all \
        --output annotate_results/depth_results/vg_test/vg_test.hdf5
    
    # psg test
    python -m osprey.eval.annotate.run_depth \
        --dataset_name psg \
        --image_data /net/nfs.cirrascale/mosaic/jamesp/Osprey/osprey/eval/psg/psg_asv2_val_test.json \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/images/coco/ \
        --output annotate_results/depth_results/psg_test/psg_asv2_val_test.hdf5

         
    '''
    parser = argparse.ArgumentParser(description='osprey sg generator', formatter_class=argparse.RawTextHelpFormatter)
    # Model config
    parser.add_argument('--dataset_name', help='datasets to use', required=True,) 
    parser.add_argument('--image_data', help='image json file', required=True) 
    parser.add_argument('--image_dir', help='path to images', required=True)
    parser.add_argument('--depth_model', help='depth anything model to use', default="depth-anything/Depth-Anything-V2-Large-hf")
    parser.add_argument('--output', help='hdf5 or directory to save results', default=None)

    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards to split the data into")
    parser.add_argument("--shard_index", type=int, default=0, help="Index of the shard to process")
    parser.add_argument("--overwrite_output_file", action="store_true", help="Overwrite the output file if it exists")

    # Dataloader
    parser.add_argument('--num_workers', help='num_workers', default=4, type=int)
    parser.add_argument('--batch_size', help='batch size', default=1000, type=int)

    args = parser.parse_args()

    # Load data
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    dataset_class = load_dataset_class(args.dataset_name)
    dataset: ImageDataset = dataset_class(image_dir=args.image_dir, image_file=args.image_data,
                                        num_shards=args.num_shards, shard_index=args.shard_index,)
    if not args.overwrite_output_file:
        dataset.filter_processed_ids(args.output)
    data_loader = DataLoader(dataset, 
                             args.batch_size, 
                             shuffle=False, 
                             drop_last=False, 
                             num_workers=args.num_workers,
                             collate_fn=lambda x: x
                            )
    logging.info(f"Number of batches: {len(data_loader)} with batch size {args.batch_size}")

    # Load OCR model    
    depth_pipeline = DepthPipeline(args.depth_model)
    depth_pipeline.process_and_save_batches(data_loader, output=args.output)