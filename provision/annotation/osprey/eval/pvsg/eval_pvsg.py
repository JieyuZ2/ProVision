import argparse
from pathlib import Path
import torch
import os
import json
import random
import numpy as np
from tqdm import tqdm
from pprint import pprint
from PIL import Image
import pandas as pd
from typing import Any, List, Dict, Tuple
from glob import glob

from sentence_transformers import SentenceTransformer, util

from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.mm_utils import tokenizer_image_token
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.train.train import DataArguments

from osprey.datasets.relation_category import RELATION_QUESTIONS, RELATION_DESCRIPTION_QUESTIONS
from osprey.datasets.psg import SG_QUESTIONS
from osprey.eval.eval import OspreyEval
from osprey.eval.psg.eval_psg import PSGEval

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

class PVSGEval(PSGEval):
    ''' Common class for loading PVSG data'''
    def __init__(
        self, 
        model_path, 
        bert_model=None,
        category=None,
        max_regions=150,
        region_mode='segmentation',
        use_object_description_context: bool=True,
        chunk_idx:int=0,
        num_chunks:int=1,
        debug=False, 
    ):
        OspreyEval.__init__(self, model_path, chunk_idx=chunk_idx, num_chunks=num_chunks, debug=debug)
        self.max_regions = max_regions

        self.region_mode = region_mode
        self.is_train = False
        self.use_object_description_context = use_object_description_context

        if category is not None:
            category_dict = json.load(open(category))
            thing_classes = category_dict['objects']['thing']
            stuff_classes = category_dict['objects']['stuff']
            self.object_classes = thing_classes + stuff_classes
            self.predicate_classes = category_dict['relations']

        #  bert model embeddings for classification
        if bert_model is not None:
            self.bert_model  = SentenceTransformer(bert_model)
            self.create_embeddings()
    
    def load_mask_image(self, mask_image_path) -> np.ndarray:
        return np.array(Image.open(mask_image_path))

    def create_segmentations(
        self, 
        pan_seg_image_path: str, 
        objects_info: list[dict[str, Any]],
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Create segmentations for objects based on the provided pan-segmentation image.

        Args:
            pan_seg_image_path (str): The file path to the pan-segmentation image.
            objects_info (list): A list of dictionaries containing information about the objects.

        Returns:
            Tuple[np.ndarray, List[int]]: A tuple containing the segmentations as a numpy array and the corresponding object IDs as a list.
        """
        im = self.load_mask_image(pan_seg_image_path)
        object_ids = []
        segmentations = []
        for object_info in objects_info:
            mask = im == object_info['object_id']
            # unlike image, not all frames in the video contain all objects
            if np.sum(mask) > 0:
                segmentations.append(mask)
                object_ids.append(object_info['object_id'])
        return np.array(segmentations, dtype=bool), object_ids


class PVSGSGEval(PVSGEval):
    ''' 
    Scene Graph Classification evaluation: Predict object labels and relationships with detected regions.
    Regions with High IoUs with Reference regions are considered in evaluation as candidates.
    '''

    def eval(self, root_path, pan_seg_path, ann_file, temperature=0.2, top_p=1.0):

        sgs = []
        ann = json.load(open(ann_file))

        data = self.get_chunk(ann['data'])

        for idx, datum in enumerate(tqdm(data)):

            video_id = str(datum['video_id'])

            frames = sorted(glob(os.path.join(root_path, '*.png')))
            
            # Generate scene graph for every frame.
            for frame in tqdm(frames):
                frame_id = Path(frame).stem

                # Load Image
                image_id = os.path.join(video_id, frame_id)
                pan_seg_image = os.path.join(pan_seg_path, f'{int(frame_id):04d}.png')
                img_path = os.path.join(root_path, f'{int(frame_id):04d}.png')
                width, height = Image.open(img_path).size

                # Load mas directly from the pan-seg image.
                objects_info: list[dict] = datum['objects']
                masks, object_ids = self.create_segmentations(pan_seg_image, objects_info)

                # Generate scene graph as list of object and relation labels. 
                object_outputs: list[str]; relation_outputs: list[str]
                object_outputs, relation_outputs = self.generate_scene_graph(img_path, torch.from_numpy(masks), temperature=temperature, top_p=top_p) 

                # Map generations into PVSG classes.
                object_labels: list[int] = self.get_object_labels(object_outputs)
                prediction_triplets: list[tuple[int,int,str]] = self.get_relation_triplets(relation_outputs) # [subj, obj, relation_str]
                predicate_names: list[str] = [triplet[2] for triplet in prediction_triplets]
                predicate_labels: list[int] = self.get_predicate_labels(predicate_names)

                # final relation names
                prediction_names: List[Tuple[int,int,str]] = []
                prediction_labels: List[Tuple[int,int,int]] = []
                for triplet, label in zip(prediction_triplets, predicate_labels):
                    name = self.predicate_classes[label]
                    prediction_names.append([triplet[0], triplet[1], name])
                    prediction_labels.append([triplet[0], triplet[1], label])

                print("[Pred]")
                print('\n'.join(object_outputs))
                print('\n'.join(relation_outputs))
                print(prediction_labels)

                sgs.append({
                    'frame_id': int(frame_id),
                    'image_id': image_id,
                    'width': width, 
                    'height': height,
                    'object_ids': object_ids,
                    'pred_raw': prediction_triplets,
                    'pred_names': prediction_names,
                    'pred_labels': prediction_labels,
                    'pred_object_names': object_outputs,
                    'pred_object_labels': object_labels,
                })

                if idx == 0:
                    pprint(sgs[0])
            
        return sgs

class PVSGHolisticEval(PVSGEval):
    ''' 
    Scene Graph Classification evaluation: Predict object labels and relationships with detected regions.
    Regions with High IoUs with Reference regions are considered in evaluation as candidates.
    '''

    def eval(self, root_path, pan_seg_path, ann_file, temperature=0.2, top_p=1.0):

        sgs = []
        ann = json.load(open(ann_file))

        data = self.get_chunk(ann['data'])

        for idx, datum in enumerate(tqdm(data)):

            video_id = str(datum['video_id'])

            frames = sorted(glob(os.path.join(root_path, '*.png')))
            
            # Process every frame of video instead.
            for frame in tqdm(frames):
                frame_id = Path(frame).stem

                # Load Image
                image_id = os.path.join(video_id, frame_id)
                pan_seg_image = os.path.join(pan_seg_path, f'{int(frame_id):04d}.png')
                img_path = os.path.join(root_path, f'{int(frame_id):04d}.png')
                width, height = Image.open(img_path).size

                # Load masks directly from the pan-seg image.
                objects_info: list[dict] = datum['objects']
                masks, object_ids = self.create_segmentations(pan_seg_image, objects_info)

                # Generate Full Scene Graph
                sg_outputs = self.generate_holistic_scene_graph(img_path, torch.from_numpy(masks), temperature=temperature, top_p=top_p)

                print("[Pred]")
                print(sg_outputs)

                print("[GT]")
                print(datum['relations'])

                sgs.append({
                    'image_id': image_id,
                    'question_id': datum['question_id'],
                    'width': width, 
                    'height': height,
                    'object_ids': object_ids,
                    'pred_raw': sg_outputs,
                })

                if idx == 0:
                    pprint(sgs[0])
                    # break
            
        return sgs

if __name__ == "__main__":
    '''
        python -m osprey.eval.pvsg.eval_pvsg_sg_cls --model  exp/multi_region_v5_gqa_cot_bs16/ \
            --max_regions 50 \
            --temperature 0.5 \
            --top_p 0.95 \
            --json $DATA/PVSG_dataset/pvsg.json \
            --img $DATA/PVSG_dataset/ego4d/frames/ec2e69c1-fd07-48ec-adff-0b2cf3ab25b6 \
            --pan_seg_img $DATA/PVSG_dataset/ego4d/masks/ec2e69c1-fd07-48ec-adff-0b2cf3ab25b6 \
            --description_context \
            --region_mode segmentation \
            --output osprey/eval/results/relation/psg_sg_cls/relation_category_interaction_description/temp0.5_top0.95_max_regions_50.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey sg evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--bert', help='path to bert model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--json', help='path to PVSG json file', 
                        default='$DATA/PVSG_dataset/pvsg.json')
    parser.add_argument('--img', help='path to frames', default='$DATA/PVSG_dataset/ego4d/frames/ec2e69c1-fd07-48ec-adff-0b2cf3ab25b6/frames')
    parser.add_argument('--pan_seg_img', help='path to masks', default='$DATA/PVSG_dataset/ego4d/frames/ec2e69c1-fd07-48ec-adff-0b2cf3ab25b6/masks')

    # Region config
    parser.add_argument('--max_regions', type=int, default=30)
    parser.add_argument('--region_mode', type=str, default='segmentation', choices=['box', 'segmentation', 'box_segmentation'])
    parser.add_argument('--description_context', action='store_true', help='Use previously generated description as context to generate relations')
    parser.add_argument('--full_sg', action='store_true', help='Generate full scene graph end to end')

    # Gen config
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print('Saving results to {}'.format(args.output))
    relation_eval = PVSGEval(args.model, 
                    max_regions=args.max_regions,        
                    bert_model=args.bert,
                    category=args.json,
                    debug=args.debug,
                    region_mode=args.region_mode,
                    use_object_description_context=args.description_context,
                    chunk_idx=args.chunk_idx,
                    num_chunks=args.num_chunks,
    )
    results = relation_eval.eval(args.img, args.pan_seg_img, args.json, args.temperature, args.top_p)

    os.makedirs(Path(args.output).parent, exist_ok=True)
    pd.DataFrame(results).to_json(args.output, orient='records', lines=True)
    print('Saved results to {}'.format(args.output))

