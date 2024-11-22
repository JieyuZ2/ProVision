import argparse
from copy import deepcopy
from pathlib import Path
import pickle
import torch
import os
import json
import numpy as np
from tqdm import tqdm
from pprint import pprint
import pandas as pd
from osprey.train.train import DataArguments

from osprey.eval.psg.run_psg import HolisticSGRunner, SGRunner, SGDataset

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

DETAILED_DESCRIPTION = "Can you provide me with a detailed description of the region in the picture marked by {}?"


class VideoSGDataset(SGDataset):
    def __init__(self, root_path: str, det_root: str):
        """
        Load from dataset from extracted frame and its detected regions

        Args:
            root_path (str): Path containing images.
            det_root (str): The path to the region data. Same image_id should be present in both the root_path and det_path.
        """
        self.data = self.load_data(root_path, det_root)
    
    def load_data(self, root_path, det_root) -> list[dict]:

        # Data Location
        # $DATA/PVSG_dataset/ego4d/frames/03f2ed96-1719-427d-acf4-8bf504f1d66d/0001.png
        # som_sam_detections/03f2ed96-1719-427d-acf4-8bf504f1d66d/som/sam_part_rle/0001.pkl 

        # Video Directories
        # root_path = '/net/nfs.cirrascale/mosaic/jamesp/data/PVSG_dataset/ego4d/frames/03f2ed96-1719-427d-acf4-8bf504f1d66d'
        # det_path = 'som_sam_detections/03f2ed96-1719-427d-acf4-8bf504f1d66d/som/sam_part_rle/'

        data = []
        video_path = root_path
        for idx, image_id in enumerate(sorted(os.listdir(video_path))):
            img_path = os.path.join(video_path, image_id)
            video_id = Path(video_path).stem
            frame_id = Path(image_id).stem
            image_id = f"{video_id}/{frame_id}"
            assert os.path.isfile(img_path), f"Image path {img_path} does not exist."

            det_path = os.path.join(det_root, frame_id + '.pkl')
            assert os.path.isfile(det_path), f"Det path {det_path} does not exist."

            data.append({
                'image_path': img_path,
                'image_id': image_id,
                'video_id': video_id,
                'frame_id': frame_id,
                'det_path': det_path,
                'index': idx
            })
        return data
    
    @classmethod
    def get_item(cls, item):

        # load detections from pickle
        detections = pickle.load(open(item['det_path'], 'rb'))

        to_return = deepcopy(item)
        to_return.update({
            'detections': detections
        })

        return to_return
    

class AllVideoSGDataset(VideoSGDataset): 

    def __init__(self, root_path: str, det_root: str, det_mode: str):
        """Initialize the RunPVSG object.

        Args:
            root_path (str): Path containing images.
            det_root (str): The path to the region data. Same image_id should be present in both the root_path and det_path.
            region_mode (str): mode of region extraction.
        """
        self.data = self.load_data(root_path, det_root, det_mode)
    
    def load_data(self, root_path, det_root, det_mode) -> list[dict]:

        # Data Location
        # $DATA/PVSG_dataset/ego4d/frames/03f2ed96-1719-427d-acf4-8bf504f1d66d/0001.png
        # som_sam_detections/03f2ed96-1719-427d-acf4-8bf504f1d66d/som/sam_part_rle/0001.pkl 

        # Video Directories
        # root_path = '/net/nfs.cirrascale/mosaic/jamesp/data/PVSG_dataset/ego4d/frames'
        # det_path = 'som_sam_detections'

        data = []
        video_ids = sorted(os.listdir(root_path))
        idx = 0
        for video_id in video_ids:
            video_path = os.path.join(root_path, video_id)
            for image_id in sorted(os.listdir(video_path)):
                img_path = os.path.join(video_path, image_id)
                frame_id = Path(image_id).stem
                image_id = f"{video_id}/{frame_id}"
                assert os.path.isfile(img_path), f"Image path {img_path} does not exist."

                # Region from detection
                det_path = os.path.join(det_root, video_id, 'som', det_mode, frame_id + '.pkl')
                assert os.path.isfile(det_path), f"Detection path to {det_path} does not exist."
                data.append({
                    'index': idx,
                    'image_path': img_path,
                    'image_id': image_id,
                    'video_id': video_id,
                    'frame_id': frame_id,
                    'det_path': det_path,
                })
                idx += 1
        return data

def load_dataset(args):
    if args.dataset == 'video':
        return VideoSGDataset(args.img, args.det)
    elif args.dataset == 'all_video':
        return AllVideoSGDataset(args.img, args.det, args.det_mode)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
         
if __name__ == "__main__":
    '''
        # SG for Predicted Regions for all videos in ego4d
        python -m osprey.eval.pvsg.run_pvsg --model exp/multi_region_v5_gqa_cot_bs16 \
            --dataset all_video \
            --img /net/nfs.cirrascale/mosaic/jamesp/data/PVSG_dataset/ego4d/frames \
            --det som_sam_detections  \
            --det_mode sam_part_rle \
            --temperature 0.5 \
            --top_p 0.95 \
            --max_regions 50 \
            --region_mode segmentation \
            --description_context \
            --output osprey/eval/results/regions/PVSG/ego4d/multi_region_v5_gqa_cot_bs16/sam_part_rle/temp0.5_top0.95_max_regions_50.jsonl
        
        # SG for specific video
        python -m osprey.eval.pvsg.run_pvsg --model exp/multi_region_v5_gqa_cot_bs16 \
            --dataset video \
            --img /net/nfs.cirrascale/mosaic/jamesp/data/PVSG_dataset/ego4d/frames \
            --det som_sam_detections/03f2ed96-1719-427d-acf4-8bf504f1d66d/som/sam_part_rle/  \
            --temperature 0.5 \
            --top_p 0.95 \
            --max_regions 50 \
            --region_mode segmentation \
            --description_context \
            --output osprey/eval/results/regions/PVSG/ego4d/multi_region_v5_gqa_cot_bs16/sam_part_rle/03f2ed96-1719-427d-acf4-8bf504f1d66d/temp0.5_top0.95_max_regions_50.jsonl
            
        # Generate Full SG Instead
        python -m osprey.eval.pvsg.run_pvsg --model exp/multi_region_v5_gqa_cot_bs16 \
            --json /data/jamesp/som/PVSG/ego4d/frames/ec2e69c1-fd07-48ec-adff-0b2cf3ab25b6/sam_seem_regions_iou_0.6_top_150.jsonl  \
            --img /net/nfs.cirrascale/mosaic/jamesp/data/PVSG_dataset/ego4d/frames/ec2e69c1-fd07-48ec-adff-0b2cf3ab25b6/ \
            --full_sg \
            --temperature 0.5 \
            --top_p 0.95 \
            --max_regions 50 \
            --region_mode segmentation \
            --description_context \
            --output osprey/eval/results/regions/PVSG/ego4d/multi_region_v5_gqa_cot_bs16/temp0.5_top0.95_max_regions_50_full_sg.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey sg evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--dataset', help='dataset to load', required=True)
    parser.add_argument('--img', help='path to imgs', required=True)
    parser.add_argument('--det', help='path to detection files', required=True)
    parser.add_argument('--det_mode', help='detection mode to get detection files', default=None)
    parser.add_argument('--full_sg', action='store_true', help='full sg generate')

    # Region config
    parser.add_argument('--max_regions', type=int, default=99)
    parser.add_argument('--region_mode', type=str, default='segmentation', choices=['box', 'segmentation', 'box_segmentation'])
    parser.add_argument('--description_context', action='store_true', help='Use previously generated description as context to generate relations')
    parser.add_argument('--long_description', action='store_true', help='Use a long description instead.')
    parser.add_argument('--no_sort_regions', action='store_true', help='Do not sort regions.')

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
    eval_class = HolisticSGRunner if args.full_sg else SGRunner

    os.makedirs(Path(args.output).parent, exist_ok=True)
    dataset = load_dataset(args)
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