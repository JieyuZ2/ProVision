import argparse
from pathlib import Path
import random
import torch
import os
import json
import numpy as np
from tqdm import tqdm
from pprint import pprint
import pandas as pd
from typing import Generator, Iterable, Iterator, List, Dict, Tuple
from PIL import Image

from torch.utils.data import Dataset

from osprey.eval.psg.eval_psg import SGEval

DETAILED_DESCRIPTION = "Can you provide me with a detailed description of the region in the picture marked by {}?"

class RunDataset(Dataset):
    def __init__(self, root_path):
        self.data = self.load_data(root_path)

    def load_data(self, root_path) -> list[dict]:
        pass

    def validate_data(data: list[dict]):
        if data is not None:
            assert isinstance(data, list), "Data should be a list."
            assert all(isinstance(item, dict) for item in data), "Data should be a list of dictionaries."
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        # if isinstance(idx, int):
        #     item: dict = self.data[idx]
        #     return self._get_item_data(item)
        # elif isinstance(idx, slice):
        #     return [self._get_item_data(item) for item in self.data[idx]]
        # else:
        #     raise TypeError(f"Invalid argument type: {type(idx)}")
    
    # Heavy loading goes here instead.
    @classmethod
    def get_item(cls, item: dict):
        return item
    
class SGDataset(RunDataset):
    def __init__(self, root_path, ann_file):
        self.data = self.load_data(root_path, ann_file)

    def load_data(self, root_path, ann_file) -> list[dict]:
        data = []
        with open(ann_file) as f:
            for idx, line in enumerate(f):
                datum: dict = json.loads(line)
                image_id = self.process_image_id(datum['image_id'])
                img_path = os.path.join(root_path, image_id + '.jpg')
                data.append({
                    'index': idx,
                    'image_id': image_id, 
                    'image_path': img_path,
                    'detections': datum['regions']
                })
        return data

    def process_image_id(self, image_id: int | str):
        return str(image_id)

class COCOSGDataset(SGDataset):
    def __init__(self, root_path, ann_file, split='train2017'):
        self.split = split
        super().__init__(root_path, ann_file) 

    def process_image_id(self, image_id: int | str) -> str:
        return os.path.join(self.split, str(image_id).zfill(12))

# Runner
class SGRunner(SGEval):
    ''' 
    Generate scene graph relationships for each object
    '''
    def __init__(self, model_path, max_regions, region_mode, 
                 use_object_description_context=True, use_long_description=False, 
                 chunk_idx=0, num_chunks=1, debug=False):
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

    @staticmethod
    def filter_processed_data(data: list[dict] | RunDataset, processed_image_ids: set) -> list[dict]:
        print('Already processed {} images'.format(len(processed_image_ids)))
        new_data = [datum for datum in data if datum['image_id'] not in processed_image_ids]
        print('Number of data to process: {} -> {}'.format(len(data), len(new_data)))
        return new_data
    
    @staticmethod
    def load_processed_ids(output_file):
        processed_image_ids = set()
        if output_file and os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    processed_image_ids.add(record['image_id'])
        return processed_image_ids
    
    def get_data_generator(self, data: list[dict] | RunDataset, output_file: str, overwrite=False) -> Iterable[list[dict]]:
        """
        Lazily yields data from the specified chunk.
        Skips preprocessed data in output_file.
        
        Args:
            data (list[dict] | RunDataset): The dataset to shard and iterate over.
            chunk_idx (int): The index of the chunk to process.
            num_chunks (int): The total number of chunks to divide the data into.

        Yields:
            Iterator over the specified chunk of data.
        """
        # Iterate over indices instead of loading the data 
        if isinstance(data, RunDataset):
            chunk = self.get_chunk(data.data)
            if overwrite:
                processed_image_ids = self.load_processed_ids(output_file)
                chunk: list[dict] = self.filter_processed_data(chunk, processed_image_ids)
            for item in tqdm(chunk, total=len(chunk)):
                yield data.get_item(item)
        else:
            chunk = self.get_chunk(data)
            if overwrite:
                processed_image_ids = self.load_processed_ids(output_file)
                chunk: list[dict] = self.filter_processed_data(chunk, processed_image_ids)
            return tqdm(chunk)
         
    @staticmethod
    def save_and_clear_results(results: list[dict], output_file):
        """
        Saves the results to the output file and clears the list

        Args:
            sgs (list[dict]): The list of dictionaries representing the results.
            output_file (str): The path to the output file.

        Returns:
            list: An empty list.
        """
        if len(results) > 0:
            with open(output_file, 'a') as f:
                for sg in results:
                    f.write(json.dumps(sg) + '\n')
                print('Saved {} batch to {}'.format(len(results), output_file))
        results.clear()     

    def preprocess(self, image: str | Image.Image, regions: list[dict], sort_regions_by_largest=True, top_k=None) -> dict:
        image = self.load_image(image)
        width, height = image.size

        boxes, segs, masks = self.create_mask_input(regions, height, width, sort_regions_by_largest=sort_regions_by_largest, top_k=top_k)
        masks: torch.Tensor = torch.from_numpy(masks)
        
        return {
            'image': image,
            'boxes': boxes,
            'segs': segs,
            'masks': masks,
        }
     
    def eval(self, data: Iterable[list[dict]], output_file: str, sort_regions_by_largest=True, temperature=0.2, top_p=1.0, batch_size=10):
        """
        Generate scene graph relationships for each object in the given data.
        Runs preprocessing on the input data and generates scene graph relationships for each object.

        Args:
            data (Iterable[list[dict]]): The input data containing information about the images and regions.
            output_file (str): The path to the output file where the results will be saved.
            sort_regions_by_largest (bool, optional): Whether to sort the regions by largest first. Defaults to True.
            temperature (float, optional): The temperature parameter for controlling the randomness of the generation process. Defaults to 0.2.
            top_p (float, optional): The top-p parameter for controlling the diversity of the generated relationships. Defaults to 1.0.
            batch_size (int, optional): The batch size for processing the data in batches. Defaults to 10.

        Returns:
            list[dict]: A list of dictionaries containing the generated scene graph relationships for each object in the input data.
        """
        results = []
        batch_results = []
        
        for datum in data:

            # Preprocess
            image_id = datum['image_id']
            image_path = datum['image_path']
            inputs = self.preprocess(image_path, datum['detections'], sort_regions_by_largest=sort_regions_by_largest, top_k=self.max_regions)
            width, height = inputs['image'].size

            # Generate scene graph for each region.
            result = self(inputs['image'], inputs['masks'], temperature=temperature, top_p=top_p)
            result.update({
                'image_id': image_id,
                'width': width, 
                'height': height,
                'pred_boxes': inputs['boxes'],
            })
            batch_results.append(result)
            results.append(result)

            # Save and clear batch results if batch size is reached
            if len(batch_results) % batch_size == 0:
                self.save_and_clear_results(batch_results, output_file)
        
        # Save and clear remaining batch_results
        if batch_results:
            self.save_and_clear_results(batch_results, output_file)
    
        return results
    
    def __call__(self, image: str | Image.Image, masks: torch.Tensor, temperature=0.2, top_p=1.0):
        """Generate object and relation labels for the given image and masks.

        Args:
            image (str | Image.Image): The input image or path to the image.
            masks (torch.Tensor): H x W tensor masks for objects in the image. 
            temperature (float, optional): The temperature parameter for sampling. Defaults to 0.2.
            top_p (float, optional): The top-p parameter for sampling. Defaults to 1.0.

        Returns:
            dict: A dictionary containing the predicted triplets and object names.
                - 'pred_triplets' (List[Tuple[int,int,str]]): The predicted triplets in the form of (subject_id, object_id, relation).
                - 'pred_object_names' (List[str]): The predicted object names.
        """

        n = len(masks)

        object_outputs, relation_outputs = self.generate_scene_graph(image, masks, temperature=temperature, top_p=top_p)
        prediction_triplets: List[Tuple[int,int,str]] = self.get_relation_triplets(relation_outputs)

        # verify triplets
        for triplet in prediction_triplets:
            assert triplet[0] < n and triplet[1] < n, 'Invalid triplet: {}'.format(triplet)
            assert triplet[0] >= 0 and triplet[1] >= 0, 'Invalid triplet: {}'.format(triplet)

        return {
            'pred_triplets': prediction_triplets, # [subj_id, obj_id, relation]
            'pred_object_names': object_outputs,  # [obj_names]
        }

class HolisticSGRunner(SGRunner):

    def __call__(self, image: str | Image.Image, masks: torch.Tensor, temperature=0.2, top_p=1.0):
        """Generate a scene graph for the given image and masks.

        Args:
            image (str | Image.Image): The input image as a file path or PIL Image object.
            masks (torch.Tensor): H x W tensor masks for objects in the image.
            temperature (float, optional): The temperature parameter for sampling. Defaults to 0.2.
            top_p (float, optional): The top-p parameter for sampling. Defaults to 1.0.

        Returns:
            dict: A dictionary containing the predicted scene graph.
        """
        sg_outputs: str = self.generate_holistic_scene_graph(image, masks, temperature=temperature, top_p=top_p)
        return {
            'pred_scene_graph': sg_outputs
        }
    
    def parse_outputs(self, outputs: str):
        try:
            if 'Relations:' not in outputs:
                return {
                    'pred_names': [],
                    'pred_labels': [],
                    'pred_object_names': [],
                    'pred_object_labels': []
            }
            object_outputs, relation_outputs = outputs.split('Relations:')
            object_outputs = object_outputs.split('Objects:')[1].strip()

            # Parse prediction
            object_outputs = [d.split(':')[1] for d in object_outputs.split('\n')]

            # Map predicate labels
            if 's:' in relation_outputs: # Handle weird case:
                if 'region1' not in relation_outputs:
                    relation_outputs = relation_outputs.replace('s:', 'region1:')
                else:
                    relation_outputs = relation_outputs.replace('s:', '')
            relation_outputs = relation_outputs.strip()
            relations = relation_outputs.split('\n')
            prediction_triplets: List[Tuple[int,int,str]] = self.get_relation_triplets(relations)
                
            return {
                'pred_triplets': prediction_triplets, # [subj_id, obj_id, relation]
                'pred_object_names': object_outputs,  # [obj_names]
            }
        except Exception:
            print('Failed to process data: ', outputs)
            return None

if __name__ == "__main__":
    '''
        python -m osprey.eval.psg.run_psg --model exp/multi_region_v5_gqa_cot_bs16 \
            --json data/regions/coco_train_2017_sam_seem_regions_iou_0.6_top_150.jsonl\
            --img $DATA/../images/coco/ \
            --max_regions 99 \
            --temperature 0.5 \
            --top_p 0.95 \
            --max_regions 99 \
            --region_mode segmentation \
            --description_context \
            --output osprey/eval/results/regions/coco_train_2017/multi_region_v5_gqa_cot_bs16/temp0.5_top0.95_max_regions_99.jsonl
        
        # Generate Full SG Instead
        python -m osprey.eval.psg.run_psg --model exp/multi_region_v3-a100 \
            --json data/regions/coco_train_2017_sam_seem_regions_iou_0.6_top_150.jsonl\
            --img ../images/coco/ \
            --full_sg \
            --max_regions 99 \
            --temperature 0.5 \
            --top_p 0.95 \
            --max_regions 99 \
            --region_mode segmentation \
            --description_context \
            --output osprey/eval/results/regions/coco_train_2017/multi_region_v3/temp0.5_top0.95_max_regions_99_full_sg.jsonl
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
    parser.add_argument('--no_sort_regions', action='store_true', help='Do not sort regions.')

    # Gen config
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.5)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=0.95)
    parser.add_argument('--output', help='path to save results to json file', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print('Saving results to {}'.format(args.output))
    eval_class = HolisticSGRunner if args.full_sg else SGRunner
    is_coco = 'coco' in args.img
    print('Data is coco: {}'.format(is_coco))

    os.makedirs(Path(args.output).parent, exist_ok=True)
    dataset_class = COCOSGDataset if is_coco else SGDataset
    dataset = dataset_class(args.img, args.json)
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
    data = run_region.get_data_generator(dataset, args.output)

    sort_regions_by_largest = not args.no_sort_regions
    results = run_region.eval(data, args.output,
        sort_regions_by_largest=sort_regions_by_largest, temperature=args.temperature, top_p=args.top_p, batch_size=args.batch_size) 
    
    # 
    # masks = run_region.preprocess(image, regions)
    # results = run_region(image, masks)