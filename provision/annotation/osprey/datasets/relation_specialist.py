import copy
import json
import pickle
from typing import List, Dict, Tuple, Literal, Union
import os
import random
from tqdm import tqdm
from enum import auto, Enum

import numpy as np
import torch
from PIL import Image
import re
import h5py
from .sg import SGDataset
from osprey.train.train import preprocess, preprocess_multimodal

''' Dataset for relationship specialist. Separate prompt is used to specify desired type of relationships (e.g. spatial, functional, etc.)'''

# Setup options as strings
class RegionSetup(Enum):
    """ Setup for region selection"""
    SUBJECT_ONLY = "subject_only"
    MENTIONED_SUBJECT = "mentioned_subject"
    MENTIONED_ALL = "mentioned_all"
    OBJECT_DETAILS = "object_details"
    OBJECT_DETAILS_ALL = "object_details_all"

RELATION_CATEGORY_QUESTIONS = {
    'emotional': [
        'What are the emotional relationships for {}?',
        'Can you describe the feelings or emotions involved in {}?',
        'How do individuals feel about each other in {}?',
        'How does an individual feel about another in {}?',
        'What emotional connections are present in {}?',
        'Identify the emotional ties in {}.',
    ],
    'functional': [
        'What are the functional relationships for {}?',
        'Can you specify the purposes or functions involved in {}?',
        'How are objects or individuals used in {}?',
        'How is an object or individual used in {}?',
        'What roles do objects or individuals play in {}?',
        'Identify the functional connections in {}.',
    ],
    'interactional': [
        'What are the interactional relationships for {}?',
        'Can you describe the interactions taking place in {}?',
        'How do individuals or objects interact in {}?',
        'How does an individual or object interact in {}?',
        'What actions are involved in {}?',
        'Identify the interactional ties in {}.',
    ],
    'social': [
        'What are the social relationships for {}?',
        'Can you outline the social connections in {}?',
        'How are individuals socially linked in {}?',
        'How is an individual socially linked in {}?',
        'What social roles are present in {}?',
        'Identify the social ties in {}.',
    ],
    'spatial': [
        'What are the spatial relationships for {}?',
        'Can you describe the physical positions involved in {}?',
        'How are objects or individuals positioned in {}?',
        'How is an object or individual positioned in {}?',
        'What spatial arrangements are present in {}?',
        'Identify the spatial connections in {}.',
    ],
    'symbolic': [
        'What are the symbolic relationships for {}?',
        'Can you explain the symbolic meanings in {}?',
        'How do objects or individuals symbolize concepts in {}?',
        'How does an object or individual symbolize a concept in {}?',
        'What metaphorical connections are present in {}?',
        'Identify the symbolic ties in {}.',
    ],
}

### Utility Functions
def shuffle_regions_and_update_mapping(
    regions: list[int], 
    region_mapping: dict[str, int]
) -> Tuple[list, dict[str, int]]:
    """
    Shuffle regions randomly and update the region_mapping with the new indices.
    Useful for training and debiasing region order.
    
    Args:
        regions (List): The list of regions to be shuffled.
        region_mapping (Dict[str, int]): The mapping from object IDs to their original region indices.
        
    Returns:
        Tuple[List, Dict[str, int]]: Shuffled regions and updated region_mapping.
    
    Example:    
        >>> regions = [0, 1, 2, 3, 4]
        >>> region_mapping = {'0': 0, '1': 1, '2': 2, '4': 4} # missing 3
        >>> shuffle_regions_and_update_mapping(regions, region_mapping)
        >>> ([4, 2, 3, 1, 0], {'4': 0, '2': 1, '1': 3, '0': 4}
    """

    random.seed(42)
    shuffled_indices = list(range(len(regions)))
    random.shuffle(shuffled_indices)
    
    # Use the shuffled indices to create a new, shuffled list of regions
    shuffled_regions = [regions[i] for i in shuffled_indices]
    
    # Create a reverse mapping from old indices to new indices
    index_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(shuffled_indices)}
    
    # Update region_mapping to reflect the new indices based on the shuffle
    updated_region_mapping = {obj_id: index_mapping[region_idx] for obj_id, region_idx in region_mapping.items()}
    
    return shuffled_regions, updated_region_mapping

def is_index(input: str | int) -> bool:
    """" Check if input is an index. """
    return str(input).isdigit()

class RelationSpecialistDataset(SGDataset):
    CLASSES = ('object',)

    def __init__(
            self,
            tokenizer,
            data_args=None,
            ann_file=None,
            region_hdf5=None,
            img_prefix=None,
            max_gt_per_img=99,
            max_relations_per_obj=20,
            
            is_train=True,
            region_mode: Literal['box', 'segmentation', 'box_segmentation']='segmentation',
            ignored_relations: List[str] = None,
            shuffle_relations: bool = False,
            add_description: bool = False,

            use_bbox_text=False,
    ):

        self.image_regions = h5py.File(region_hdf5, 'r') # {'image_id': {'regions': [], 'scene_graph_regions': []}}
        self.add_description = add_description
        self.valid_region_setups = None
        self.default_region_setup = None
        self.setup_valid_proposals()

        super().__init__(tokenizer, data_args, ann_file, img_prefix, 
                         max_gt_per_img=max_gt_per_img, max_relations_per_obj=max_relations_per_obj,
                         is_train=is_train, region_mode=region_mode, ignored_relations=ignored_relations,
                         shuffle_relations=shuffle_relations, use_bbox_text=use_bbox_text
                         )
    
    def setup_valid_proposals(self):
        self.valid_region_setups = [
            # RegionSetup.SUBJECT_ONLY,
            RegionSetup.MENTIONED_SUBJECT,
            RegionSetup.MENTIONED_ALL,
            RegionSetup.OBJECT_DETAILS,
            RegionSetup.OBJECT_DETAILS_ALL
        ]
        self.default_region_setup = RegionSetup.OBJECT_DETAILS
        
        # self.valid_region_setups = [1,2,3,4]
        # self.default_region_setup = 3

    def get_region_proposal_setup(self):
        """
        Gets the region proposal setup.
        
        Returns:
            int: The region proposal setup.
        """
        if self.is_train:
            return random.choice(self.valid_region_setups)
        return self.default_region_setup

    def get_region(self, regions_dict: dict[str, list], region_id):
        # regions = self.image_regions[str(image_id)]
        # if is_index(region_id):
        #     region = pickle.loads(regions['regions'][()])[int(region_id)]
        # elif isinstance(region_id, str):
        #     prefix, _, index = region_id.rpartition('_')
        #     region = pickle.loads(regions[f'{prefix}_regions'][()])[int(index)]
        if is_index(region_id):
            region = regions_dict['regions'][int(region_id)]
        elif isinstance(region_id, str):
            prefix, _, index = region_id.rpartition('_')
            region = regions_dict[f'{prefix}_regions'][int(index)]
        return region
    
    def get_regions_dict(self, image_id) -> dict[str, list]:
        return pickle.loads(self.image_regions[image_id][()])
        
    def get_region_proposals(self, ann, subj: str, subject_relations, setup: RegionSetup, regions_dict):
        """
        Selects the regions based on the provided setup configuration.

        Args:
            ann (Annotation): The annotation object containing relevant data.
            subj (List[str]): A list of subject strings to be processed.
            subject_relations (Dict[str, Any]): A dictionary mapping subjects to their relations.
            setup (int): The setup configuration object that defines how regions should be selected.

        Returns:
            Tuple[List[Region], Dict[str, Region]]: A tuple containing:
                - A list of selected Region objects.
                - A dictionary mapping subject strings to their corresponding Region objects.
        """
        
        # Setup: Only the subject (useful for description class)
        if setup == RegionSetup.SUBJECT_ONLY:
            if isinstance(subj, str):
                object_ids = [subj]
            else:
                object_ids = subj

        # Setup: Only the ones mentioned in subject-object relations for the given subject
        elif setup == RegionSetup.MENTIONED_SUBJECT:
            object_ids = subject_relations['mentioned_objects'] 

        # Setup: Only the ones mentioned in subject-object relations for all subjects
        elif setup == RegionSetup.MENTIONED_ALL:
            object_ids = ann['all_mentioned_objects']
    
        # Setup: Regions in object_details.
        elif setup == RegionSetup.OBJECT_DETAILS:
            object_ids = ann['object_details'].keys()

        # Setup: Regions in object_details and all regions (excluding scene graph regions)
        elif setup == RegionSetup.OBJECT_DETAILS_ALL:
            object_ids = ann['object_details'].keys()

        else:
            raise ValueError("Setup must be one of the RegionSetup enum values. Found: {}".format(setup))
        
        # Get region candidates based on the setup
        id_region_mapping: dict[str, str|int] = ann['id_region_mapping']
        regions = [] # list of regions
        region_mapping: dict[str, int] = {} # maps object_id to the index of the regions
        for obj_id in object_ids:
            region_id = id_region_mapping[obj_id]
            region = self.get_region(regions_dict, region_id)
            region_index = len(regions)
            region_mapping[obj_id] = region_index
            regions.append(region)
            if len(regions) >= self.max_gt_per_img:
                break

        if setup == RegionSetup.OBJECT_DETAILS_ALL: # Add uncovered regions
            indices = [id_region_mapping[obj_id] for obj_id in object_ids if is_index(id_region_mapping[obj_id])]
            uncovered_regions = [region for idx, region in enumerate(regions_dict['regions']) if idx not in indices]
            regions += uncovered_regions
        
        # Cut off max regions
        regions = regions[:self.max_gt_per_img]
        assert len(region_mapping) <= self.max_gt_per_img

        if any([v >= len(regions) for v in region_mapping.values()]):
            print('Warning: The number of regions in the mapping exceeds the maximum allowed.')
        
        return regions, region_mapping

    def load_annotations(self, ann_file):
        """
        Annotation Format:
            image_id                                                             61522
            width                                                                  910
            height                                                                1024
            relations                {'1': {'description': 'A black coffee cup with...
            object_details           {'0': {'bbox': [316, 0, 634, 214], 'text': ['[...
            id_region_mapping        {'0': 8, '1': 2, '2': 13, '3': 2, '4': 0, '5':...
            regions                  [{'segmentation': {'size': [1024, 910], 'count...
            all_mentioned_objects     [0, 1, 2, 4, 8, 7, 97, 6, 9, 64, 3, 104, 108, 5]
        """
        ann_list = []
        with open(ann_file, 'r') as f:
            for line in f:  
                ann_list.append(json.loads(line))

        data_infos = []
        for idx,ann in enumerate(tqdm(ann_list)):
            image_id = str(ann['image_id'])
            img_path = os.path.join(self.img_prefix, image_id+'.jpg')
            regions_dict: dict[str, list] = self.get_regions_dict(image_id)
            for subj, subject_relations in  ann['relations'].items():

                # Get Region Proposals
                setup = self.get_region_proposal_setup()
                regions, region_mapping = self.get_region_proposals(ann, subj, subject_relations, setup, regions_dict)
                if self.is_train:
                    regions, region_mapping = shuffle_regions_and_update_mapping(regions, region_mapping)
                
                # Get segmentations
                boxes, segmentations =  self.process_regions(regions)
                if len(boxes) == 0:
                    continue
                assert len(boxes) == len(segmentations), \
                    "number of boxes: ({}) and segmentations: ({}) should match".format(len(boxes), len(segmentations))

                # Create relation conversation
                convs: list[dict] = self.generate_conversation(subj, subject_relations['relations'], region_mapping)
                if convs:
                    data_infos.append(dict(
                        img_path = img_path,
                        boxes = boxes, 
                        segmentations=segmentations,
                        region_mapping=region_mapping,
                        convs = convs
                    ))
                else:
                    print(f'No relations found for subject {subj} , image {image_id}, index {idx}')
        
        return data_infos

    def generate_conversation(self, subj: str, relations: dict[str, list], region_mapping: dict[str, int]) -> List:
        """
            'relations': {
                'relation_category': [['obj_id', 'relation_name'], ...],
            }
        """
        try:
            sg_s = []
            """ Generate categorized relations """
            for relation_category, relation_list in relations.items():
                relations = relation_list[:self.max_relations_per_obj]
                if len(relations) == 0:
                    continue

                rel_text = []
                relations = sorted(relations, key=lambda x: self.get_region_id(x[0], region_mapping))
                for relation in relations:
                    obj_id, rel_name = relation
                    region_obj = self.get_region_id(obj_id, region_mapping)
                    rel_text.append(f"region{region_obj} {rel_name}")
                a = ', '.join(rel_text)

                region_subj: int = self.get_region_id(subj, region_mapping)
                q = random.choice(RELATION_CATEGORY_QUESTIONS[relation_category])
                q = q.format(f'region{region_subj}')
                sg_s.append({'from': 'human', 'value': q})
                sg_s.append({'from': 'gpt', 'value': a})

            return sg_s
    
        except KeyError:
            return None

if __name__ == '__main__':
    from transformers import AutoTokenizer, CLIPImageProcessor
    from types import SimpleNamespace
    from osprey.model.multimodal_encoder.clip_encoder import get_image_processor
    from ..visualize.utils import draw_segmentation

    tokenizer = AutoTokenizer.from_pretrained('../models/Osprey-7b/')
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')
    dataset = RelationSpecialistDataset(
        tokenizer, data_args=data_args, 
        region_hdf5='data/relation/regions/train_coco_relation_category_interaction_sam_seem_regions_150_verified_qwen_llava_rule_clean.hdf5',
        ann_file='data/relation/train_coco_relation_category_interaction_sam_seem_regions_150_verified_qwen_llava_rule_clean.jsonl',
        img_prefix="/net/nfs.cirrascale/mosaic/jamesp/images/gqa/images",
        region_mode='box_segmentation',
        is_train=False,
        add_description=True,
    )
    draw_segmentation(dataset, idx=0, output_image='vg_sg.jpg')
    breakpoint()
    print(dataset.data_infos[1]['convs'][1]['value'])

    # check if length mask test fails.

    bad_id = [] 
    for idx in tqdm(range(len(dataset))):
        try:
            data = dataset.__getitem__(idx)
            cur_input_ids = data['input_ids']
            masks = data['masks']
            mask_idx = torch.nonzero(cur_input_ids==dataset.tokenizer.convert_tokens_to_ids(['<mask>'])[0])
            if len(masks) != len(mask_idx):
                print('not matching', idx)
                bad_id.append(idx)
        except Exception:
            bad_id.append(idx)
    print('bad ids: {}'.format(bad_id))
    breakpoint()
