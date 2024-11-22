import copy
import json
from typing import List, Dict, Tuple, Literal, Union
import os
import random
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image
import re

from .sg import SGDataset

RELATION_QUESTIONS = [
    'Generate list of relationships for: {}.',
    'Assign relations for: {}',
    'Can you assign relations to objects in {}?',
    'How can we map all relationships for {}?',
    'What are the all connections you see in {}?',
    'Identify the inter-regional relationships for {}.',
    'Could you detail the interactions for {}?',
    'Please outline the network of relationships for {}.',
    'Can you delineate the ties binding {}?',
    'Could you classify the types of relationships present in {}?',
]

RELATION_DESCRIPTION_QUESTIONS = [
    'Generate a description for: {}.',
    'Describe {} in details.',
    'What is going on in {}?',
]

RELATION_SUMMARY_QUESTIONS = [
    'Provide a dense summary including all details in the image.',
    'Provide a dense localized narrative of this entire image.',
    'Generate a comprehensive summary of the image.',
]

### Utility Functions
def shuffle_regions_and_update_mapping(regions: List, region_mapping: Dict[str, int]) -> Tuple[List, Dict[str, int]]:
    """
    Shuffles the regions list and updates the region_mapping to reflect the new indices of the regions.
    Useful for training and debiasing region index.
    
    Args:
        regions (List): The list of regions to be shuffled.
        region_mapping (Dict[str, int]): The mapping from object IDs to their original region indices.
        
    Returns:
        Tuple[List, Dict[str, int]]: The shuffled list of regions and the updated region mapping.
    
    Example:    
        >>> regions = [0, 1, 2, 3, 4]
        >>> region_mapping = {'0': 0, '1': 1, '2': 2, '4': 4} # missing 3
        >>> shuffle_regions_and_update_mapping(regions, region_mapping)
        >>> ([4, 2, 3, 1, 0], {'4': 0, '2': 1, '1': 3, '0': 4}
    """
    shuffled_indices = list(range(len(regions)))
    random.shuffle(shuffled_indices)
    
    # Use the shuffled indices to create a new, shuffled list of regions
    shuffled_regions = [regions[i] for i in shuffled_indices]
    
    # Create a reverse mapping from old indices to new indices
    index_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(shuffled_indices)}
    
    # Update region_mapping to reflect the new indices based on the shuffle
    updated_region_mapping = {obj_id: index_mapping[region_idx] for obj_id, region_idx in region_mapping.items()}
    
    return shuffled_regions, updated_region_mapping

class RelationCategoryDataset(SGDataset):
    CLASSES = ('object',)

    def __init__(
            self,
            tokenizer,
            data_args=None,
            ann_file=None,
            img_prefix=None,
            max_gt_per_img=99,
            max_relations_per_obj=20,
            
            is_train=True,
            region_mode: Literal['box', 'segmentation', 'box_segmentation']='segmentation',
            ignored_relations: List[str] = None,
            num_epoch: int=1,
            shuffle_relations: bool = False,
            add_description: bool = False,

            use_bbox_text=False,
    ):

        # FIXME: Temporary hack to train with multiple epochs
        self.num_epoch = num_epoch
        self.add_description = add_description

        self.valid_region_setups = None
        self.default_region_setup = None
        self.setup_valid_proposals()

        super().__init__(tokenizer, data_args, ann_file, img_prefix, 
                         max_gt_per_img=max_gt_per_img, max_relations_per_obj=max_relations_per_obj,
                         is_train=is_train, region_mode=region_mode, ignored_relations=ignored_relations,
                         shuffle_relations=shuffle_relations, use_bbox_text=use_bbox_text
                         )


        print('{} (num epochs {}): {}'.format(self.__class__.__name__, num_epoch, len(self.data_infos)))
    
    """
        Annotation Format:
            image_id                                                             61522
            width                                                                  910
            height                                                                1024
            relations                {'1': {'description': 'A black coffee cup with...
            object_details           {'0': {'bbox': [316, 0, 634, 214], 'text': ['[...
            id_region_mapping        {'0': 8, '1': 2, '2': 13, '3': 2, '4': 0, '5':...
            regions                  [{'segmentation': {'size': [1024, 910], 'count...
            scene_graph_regions      [{'object_id': '615220', 'segmentation': {'siz...
            all_mentioned_objects     [0, 1, 2, 4, 8, 7, 97, 6, 9, 64, 3, 104, 108, 5]
    """

    def setup_valid_proposals(self):
        self.valid_region_setups = [1,2,3,4]
        self.default_region_setup = 3

    def get_region_proposal_setup(self):
        """
        Gets the region proposal setup.
        
        Returns:
            int: The region proposal setup.
        """
        if self.is_train:
            return random.choice(self.valid_region_setups)
        return self.default_region_setup
        

    def get_region_proposals(self, ann, subj: Union[str, List[str]], subject_relations, setup: int):
        """
        Gets the region proposals.
        
        Args:
            ann: The annotation.
            subj: List of strings
            subject_relations: The subject relations.
            setup: The setup.
            
        Returns:
            Tuple: The regions and region mapping.
        """

        id_region_mapping = ann['id_region_mapping']

        def is_index(text: Union[str, int]):
            return str(text).isdigit()

        def get_regions(object_ids) -> Tuple[List, Dict[str, int]]:
            regions = []
            region_mapping = {}
            for obj_id in object_ids:
                index = id_region_mapping[obj_id]
                if is_index(index):
                    region = ann['regions'][int(index)]
                elif isinstance(index, str):
                    prefix, _, index = index.rpartition('_')
                    region = ann[f'{prefix}_regions'][int(index)]
                
                region_index = len(regions)
                region_mapping[str(obj_id)] = region_index
                regions.append(region)

                if len(regions) >= self.max_gt_per_img:
                    break

            return regions, region_mapping 

        # Setup 0: Only the subject (useful for description class)
        if setup == 0:
            if isinstance(subj, str):
                object_ids = [subj]
            else:
                object_ids = subj

        # Setup 1: Only the ones mentioned in subject-object relations for the given subject
        elif setup == 1:
            object_ids = subject_relations['mentioned_objects'] 

        # Setup 2: Only the ones mentioned in subject-object relations for all subjects
        elif setup == 2:
            object_ids = ann['all_mentioned_objects']
    
        # Setup 3: Regions in object_details.
        elif setup == 3:
            object_ids = ann['object_details'].keys()

        # Setup 4: Regions in object_details and all regions (excluding scene graph regions)
        elif setup == 4:
            object_ids = ann['object_details'].keys()

        else:
            raise ValueError("Setup must be in range 0 to 3. Found: {}".format(setup))
        
        regions, region_mapping = get_regions(object_ids)

        if setup == 4: # Additionally add region proposals
            indices = [id_region_mapping[obj_id] for obj_id in object_ids if is_index(id_region_mapping[obj_id])]
            uncovered_regions = [region for idx, region in enumerate(ann['regions']) if idx not in indices]
            regions += uncovered_regions
        
        # Cut off max regions
        regions = regions[:self.max_gt_per_img]
        assert len(region_mapping) <= self.max_gt_per_img

        if any([v >= len(regions) for v in region_mapping.values()]):
            print('region mapping is too much...')
        
        return regions, region_mapping

    def load_annotations(self, ann_file):

        ann_list = []
        for _ in range(self.num_epoch):
            with open(ann_file, 'r') as f:
                for line in f:  
                    ann_list.append(json.loads(line))

        data_infos = []
        for idx,ann in enumerate(tqdm(ann_list)):

            image_id = str(ann['image_id'])

            img_path = os.path.join(self.img_prefix, image_id+'.jpg')

            for subj, relation_annot in  ann['relations'].items():
        
                # Get Region Proposals
                setup = self.get_region_proposal_setup()
                regions, region_mapping = self.get_region_proposals(ann, subj, relation_annot, setup)

                if self.is_train:
                    regions, region_mapping = shuffle_regions_and_update_mapping(regions, region_mapping)
                
                # Get Segmentation Information from regions
                boxes, segmentations =  self.process_regions(regions)
                
                if len(boxes) == 0:
                    continue
                
                assert len(boxes) == len(segmentations), \
                    "number of boxes: ({}) and segmentations: ({}) should match".format(len(boxes), len(segmentations))

                # Create relation conversation
                # Note: `self.begin_string` is added as prefix in __getitem__ 
                sg_s = self.generate_conversation(subj, relation_annot, region_mapping)
                if sg_s:
                    data_infos.append(dict(
                        img_path = img_path,
                        boxes = boxes, 
                        segmentations=segmentations,
                        region_mapping=region_mapping,
                        convs = sg_s
                    ))
                else:
                    print(f'No relations found for subject {subj} , image {image_id}, index {idx}')
        
        return data_infos

    def generate_conversation(self, subj, relation_annot, region_mapping) -> List:
        try:
            sg_s = []
            region_subj = self.get_region_id(subj, region_mapping)
            q = random.choice(RELATION_QUESTIONS).format(f'region{region_subj}')

            ''' Get the relation triplet'''
            relations = []
            for category, relation_list in relation_annot['relations'].items():
                relations += relation_list
            relation_set = {tuple(relation) for relation in relations}
            relations = [list(relation) for relation in relation_set]
            relations: List[List[str, str]] = relations[:self.max_relations_per_obj]

            ''' Convert triplet to response'''
            rel_text = []
            relations = sorted([relation for relation in relations], key=lambda x: self.get_region_id(x[0], region_mapping))
            for relation in relations:
                obj_id, rel_name = relation
                region_obj = self.get_region_id(obj_id, region_mapping)
                rel_text.append(f"region{region_obj} {rel_name}")
            a = ', '.join(rel_text)


            if self.add_description: # add prompt for getting description of object.
                q_description = random.choice(RELATION_DESCRIPTION_QUESTIONS).format(f'region{region_subj}')
                a_description = relation_annot['description']
                sg_s.append({'from': 'human', 'value': q_description})
                sg_s.append({'from': 'gpt', 'value': a_description})
                sg_s.append({'from': 'human', 'value': q})
                sg_s.append({'from': 'gpt', 'value': a})

            else:
                sg_s.append({'from': 'human', 'value': q})
                sg_s.append({'from': 'gpt', 'value': a})

            return sg_s
    
        except KeyError:
            return None

class RelationDescriptionDataset(RelationCategoryDataset):
 
    def setup_valid_proposals(self):
        self.valid_region_setups = [0,2,3,4]
        self.default_region_setup = 0
    
    def load_annotations(self, ann_file):

        ann_list = []
        for _ in range(self.num_epoch):
            with open(ann_file, 'r') as f:
                for line in f:  
                    ann_list.append(json.loads(line))

        data_infos = []
        for idx,ann in enumerate(tqdm(ann_list)):

            image_id = str(ann['image_id'])

            img_path = os.path.join(self.img_prefix, image_id+'.jpg')

            subjects, relation_annots = zip(*ann['relations'].items())
            descriptions = [relation_annot['description'] for relation_annot in relation_annots]

            # Get regions
            setup = self.get_region_proposal_setup()
            regions, region_mapping = self.get_region_proposals(ann, subjects, None, setup)
            if self.is_train:
                regions, region_mapping = shuffle_regions_and_update_mapping(regions, region_mapping)
            boxes, segmentations =  self.process_regions(regions)
            if len(boxes) == 0:
                continue
            assert len(boxes) == len(segmentations), \
                "number of boxes: ({}) and segmentations: ({}) should match".format(len(boxes), len(segmentations))

            # Create relation conversation
            # Note: `self.begin_string` is added as prefix in __getitem__ 
            sg_s = self.generate_conversation(subjects, descriptions, region_mapping)
            if sg_s:
                data_infos.append(dict(
                    img_path = img_path,
                    boxes = boxes, 
                    segmentations=segmentations,
                    region_mapping=region_mapping,
                    convs = sg_s)
                )
        
        return data_infos

    def generate_conversation(self, subjects, descriptions, region_mapping) -> List:
        try:
            sg_s = []
            for idx, (subj, description) in enumerate(zip(subjects, descriptions)):
                region_subj = self.get_region_id(subj, region_mapping)
                q = random.choice(RELATION_DESCRIPTION_QUESTIONS).format(f'region{region_subj}')
                a = description

                sg_s.append({'from': 'human', 'value': q})
                sg_s.append({'from': 'gpt', 'value': a})

            return sg_s

        except KeyError:
            return None


class RelationSummaryDataset(RelationCategoryDataset):

    def setup_valid_proposals(self):
        self.valid_region_setups = [3,4]
        self.default_region_setup = 3

    def load_annotations(self, ann_file):

        ann_list = []
        for _ in range(self.num_epoch):
            with open(ann_file, 'r') as f:
                for line in f:  
                    ann_list.append(json.loads(line))

        data_infos = []
        for idx,ann in enumerate(tqdm(ann_list)):

            image_id = str(ann['image_id'])

            img_path = os.path.join(self.img_prefix, image_id+'.jpg')

            # Get Region Proposals
            setup = self.get_region_proposal_setup()
            regions, region_mapping = self.get_region_proposals(ann, None, None, setup)
            
            # Get Segmentation Information from regions
            boxes, segmentations =  self.process_regions(regions)
            
            if len(boxes) == 0:
                continue
            
            assert len(boxes) == len(segmentations), \
                "number of boxes: ({}) and segmentations: ({}) should match".format(len(boxes), len(segmentations))

            # Create relation conversation
            sg_s = self.generate_conversation(ann['summary'], region_mapping)
            if sg_s:
                data_infos.append(dict(
                    img_path = img_path,
                    boxes = boxes, 
                    segmentations=segmentations,
                    convs = sg_s)
                )
            else:
                print(f'No relations found for image {image_id} and index {idx}')

        return data_infos
    
    def generate_conversation(self, summary: str, region_mapping: Dict[str, int]) -> List:
        try:
            sg_s = []
            q = random.choice(RELATION_SUMMARY_QUESTIONS)
            a = self.map_summary_with_regions(summary, region_mapping) 
            sg_s.append({'from': 'human', 'value': q})
            sg_s.append({'from': 'gpt', 'value': a})

            return sg_s
        except KeyError:
            return None

    def map_summary_with_regions(self, summary: str, region_mapping: Dict[str, int]):

        # Pattern to find descriptions and object ids
        pattern = re.compile(r'<([^>]+)>\s*\[([^\]]+)\]')
        
        def replace_with_region(match):
            description = match.group(1)  # The descriptive text inside <>
            obj_ids = match.group(2).split(', ')  # The object ids as a list of strings
            
            # Map object ids to region ids and add 1 for 1-indexing
            region_ids = [f"region{region_mapping[obj_id]+1}" for obj_id in obj_ids if obj_id in region_mapping]
            
            # Join all region ids with ', ' and combine with the description
            if region_ids:  # Ensure there are mapped regions
                return f"|{description}| at " + f"[{', '.join(region_ids)}]"
            else:
                return description  # Return description without regions if no mapping found

        # Replace all matches in the summary using the replace_with_region function
        mapped_summary = pattern.sub(replace_with_region, summary)
        
        return mapped_summary


if __name__ == '__main__':
    from transformers import AutoTokenizer, CLIPImageProcessor
    from osprey.model.multimodal_encoder.clip_encoder import get_image_processor
    from types import SimpleNamespace
    import cv2
    import supervision as sv

    def draw_segmentation(idx: int, output_image='vg_sg.jpg'):
        info =  dataset.data_infos[idx]
        data = dataset.__getitem__(idx, debug=True)
        sg = info['convs']
        img_path = info['img_path']
        im = cv2.imread(img_path)
        boxes = np.array(info['boxes'])
        mask = np.array(data['masks'].numpy(), dtype=bool)
        ids = np.array(range(1,len(mask)+1))
        labels = [f"[{idx}]" for idx in ids]
        detections = sv.Detections(xyxy=boxes, mask=mask, class_id=ids)
        box_annotator = sv.BoxAnnotator()
        annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

        annotated_image = im.copy()
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        cv2.imwrite(output_image,annotated_image)

    tokenizer = AutoTokenizer.from_pretrained('../models/Osprey-7b/')
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')
    dataset = RelationCategoryDataset(
        tokenizer, data_args=data_args, 
        # ann_file='data/relation/train_coco_summary_sam_seem_regions_150.jsonl',
        ann_file='data/relation/train_coco_relation_category_interaction_sam_seem_regions_150_verified_qwen_llava_rule.jsonl',
        # img_prefix="../images/gqa/images",
        img_prefix="../images/gqa/images",
        region_mode='box_segmentation',
        is_train=False,
        add_description=True,
        use_bbox_text=True
    )
    draw_segmentation(0)
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
