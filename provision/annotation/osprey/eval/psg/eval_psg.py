import os
import json
import random
import re
import traceback
from typing import List, Literal, Dict, Tuple, Generator, Union
from osprey.train.train import DataArguments
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.mm_utils import tokenizer_image_token
from osprey.datasets.multi_region import MultiRegionDataset
from osprey.datasets.prompts import RELATION_QUESTIONS, RELATION_DESCRIPTION_QUESTIONS, SG_QUESTIONS
from osprey.eval.eval import OspreyEval

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util

import torch
from torchvision.ops.boxes import box_iou

import cv2
import logging
from panopticapi.utils import rgb2id
import numpy as np

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

def get_attributes(object_name):
    return [n.strip() for n in object_name.split('-')[1:] if n not in ['stuff','other', 'merged'] and n.strip()]

def obj_to_query(query: str) -> str:
    object_name = query.split('-')[0]
    attributes = get_attributes(query)
    if len(attributes) > 0:
        attributes = ', '.join(attributes)
        return f"this object is {object_name} in {attributes}"
    else:
        return f"this object is {object_name}"

def pred_to_query(query: str) -> str:
    return f"object is {query} another object"

def get_region_id(region_str: str) -> int:
    """
    Extracts and converts the region ID from a string to 0-indexed integer.
    """
    match = re.search(r'region(\d+)', region_str)
    if match:
        return int(match.group(1)) - 1
    return -1

class SceneGraphParser:
    """
    Class for parsing scene graph outputs and assigning class labels. 
    SBert model is used to measure the similarity between generated outputs and class labels.
    Then, the closest class label is assigned to the generated output.
    """
    def __init__(
        self, 
        bert_model=None,
        category=None,
        device='cuda'
    ):

        if category is not None:
            category_dict = json.load(open(category))
            self.object_classes = category_dict['classes']
            self.predicate_classes = category_dict['predicate_classes']

        # bert model embeddings for classification
        self.device = device
        if bert_model is not None:
            self.bert_model  = SentenceTransformer(bert_model, device=device)
            self.create_embeddings()
    
    def parse_outputs(self, outputs: str, sg_mode: int = 2, top_k: int=None, get_labels=True) -> dict[str, list]:
        """ 
        Parse generated scene graph outputs into object and relation predictions. 
        If top_k is specified, only top_k relations are returned.
        """
        try:
            result = {
                'pred_object_names': [], # list[str]
                'pred_triplets': [],    # list[Tuple[int,int,str]]
            }
            if get_labels:
                result.update({
                    'pred_object_labels': [], # list[int]
                    'pred_names': [],     # list[Tuple[int,int,str]]
                    'pred_labels': [],    # list[Tuple[int,int,int]]
                })
            if 'Relations:' not in outputs:
                return result
            
            assert sg_mode in [0, 1, 2], "Invalid scene graph mode. Choose from [0, 1, 2]."

            object_outputs, relation_outputs = outputs.split('Relations:')
            object_outputs = object_outputs.split('Objects:')[1].strip()

            # Parse object prediction
            object_outputs = [d.split(':')[1].strip() for d in object_outputs.split('\n')]

            # Map predicate labels
            # Handle weird case:
            if 's:' in relation_outputs:
                if 'region1' not in relation_outputs:
                    relation_outputs = relation_outputs.replace('s:', 'region1:')
                else:
                    relation_outputs = relation_outputs.replace('s:', '')
            relation_outputs = relation_outputs.strip()
            relations = relation_outputs.split('\n')
            prediction_triplets: List[Tuple[int,int,str]] = self.get_relation_triplets(relations, sg_mode)
            
            result['pred_triplets'] = prediction_triplets
            result['pred_object_names'] = object_outputs

            # Get object and predicate labels for eval
            if get_labels:
                object_labels: List[int] = self.get_object_labels(object_outputs)
                predicate_names = [triplet[2] for triplet in prediction_triplets]
                predicate_labels = self.get_predicate_labels(predicate_names)

                # Gather relation triplets
                prediction_names: List[Tuple[int,int,str]] = []
                prediction_labels: List[Tuple[int,int,int]] = []
                for triplet, label in zip(prediction_triplets, predicate_labels):
                    name = self.predicate_classes[label]
                    subj = triplet[0]
                    obj = triplet[1]

                    prediction_names.append([subj, obj, name])
                    prediction_labels.append([subj, obj, label])
                
                # For now, we randomly sample top_k
                if top_k and len(prediction_labels) > top_k:
                    random.shuffle(prediction_labels)
                    random.shuffle(prediction_names)
                    prediction_labels = prediction_labels[:top_k]
                    prediction_names = prediction_names[:top_k]
                
                result.update({
                    'pred_names': prediction_names,
                    'pred_labels': prediction_labels,
                    'pred_object_labels': object_labels
                })
                
            return result
                
            
        except ValueError as e:
            print('Failed to process scene graph output: ', outputs)
            traceback.print_exc()
            return None
        
    def get_bbox_coverage(self, gt_boxes: List[List[int]], pred_boxes: List[List[int]], threshold: float = 0.5) -> float:
        # Returns the coverage of predicted bounding boxes with respect to ground truth bounding boxes.
        # Coverage is the mean IoU of predicted bounding boxes with respect to ground truth bounding boxes.
        # Your implementation here:
        if len(pred_boxes) == 0:
            return 0.0
        if len(gt_boxes) == 0:
            return 1.0
        gt_boxes = torch.tensor(gt_boxes) # M x 4
        pred_boxes = torch.tensor(pred_boxes) # N x 4
        iou = box_iou(gt_boxes, pred_boxes) # M x N

        gt_iou = iou.max(dim=1).values
        return (gt_iou > threshold).float().mean().item()
        
        
    def create_embeddings(self):
        """
        Creates embeddings for object and predicate classes using BertModel.
        """
        object_queries = [obj_to_query(c) for c in self.object_classes]
        predicate_queries = [pred_to_query(c) for c in self.predicate_classes]
        self.object_embeddings = self.bert_model.encode(object_queries, convert_to_tensor=True, device=self.device)
        self.predicate_embeddings = self.bert_model.encode(predicate_queries, convert_to_tensor=True, device=self.device)
    
    def get_object_labels(self, object_outputs: List[str]) -> List[int]:
        """ Returns object label indices for given object predictions. """
        if isinstance(object_outputs, str):
            object_outputs = [object_outputs]
        labels = [obj_to_query(p) for p in object_outputs]
        cur_label_emb = self.bert_model.encode(labels, convert_to_tensor=True, device=self.device)
        pred_logits: torch.Tensor = util.cos_sim(cur_label_emb, self.object_embeddings)

        values, indices  = pred_logits.max(1)
        return [index.item() for index in indices]

    def get_predicate_labels(self, predicate_outputs: List[str]) -> List[int]:
        """ Returns predicate label indices for given predicate predictions. """
        if isinstance(predicate_outputs, str):
            predicate_outputs = [predicate_outputs]

        labels = [pred_to_query(p) for p in predicate_outputs]
        if len(labels) == 0:
            return []
        cur_label_emb = self.bert_model.encode(labels, convert_to_tensor=True, device=self.device)
        pred_logits: torch.Tensor = util.cos_sim(cur_label_emb, self.predicate_embeddings)

        values, indices  = pred_logits.max(1)
        return [index.item() for index in indices]

    def get_relation_triplets(self, relations: List[str], sg_mode: int) -> List[Tuple[int, int, str]]:
        """Parse generated relation string into a list of triplets (sub_id, obj_id, pred_name)"""

        if sg_mode == 0:
            return self.get_relation_triplets_mode0(relations)
        else:
            return self.get_relation_triplets_mode2(relations)
    
    def get_relation_triplets_mode0(self, relations: List[str]) -> List[Tuple[int, int, str]]:
        def get_triplet(relation_str: str) -> Generator[int, int, str]:
            """
            Extracts the triplet (source, relation, target) from a relation string.

            Generated output has the format: 'region<id>: relation_name: obj1_, obj2, ...; ...'. 
            """
            subject, relation_list = relation_str.split(':', 1)
            source_id: int = get_region_id(subject)

            delimiter_rel = ';'
            delimiter_obj = ','
            for relation in relation_list.split(delimiter_rel):
                relation = relation.strip()
                if ':' not in relation:
                    continue
                try:
                    relation_type, objects = relation.split(':', 1)
                    objects = objects.split(delimiter_obj)
                    for obj in objects:
                        target_id = get_region_id(obj)
                        yield (source_id, target_id, relation_type)
                except ValueError as e:
                    print('Failed to parse relation: {}'.format(relation))
                    continue
        triplets = []
        for rel in relations:
            for triplet in get_triplet(rel):
                triplets.append(triplet)
        return triplets

    def get_relation_triplets_mode2(self, relations: List[str]) -> List[Tuple[int, int, str]]:
        def get_triplet(relation_str: str) -> Generator[int, int, str]:
            """
            Extracts the triplet (source, relation, target) from a relation string.

            Update [4/3/24]:
                Generated output has the format: 'region<id> relation, ...'. 
                This is useful for getting relation calibration score for given object_id.
            """
            parts = relation_str.split(':')
            source_id: int = get_region_id(parts[0])

            # Find all occurrences of patterns like "region{id} relation"
            if ',' in parts[1]:
                # Handling multiple relations for the same source region
                relations = parts[1].split(',')
                for relation in relations:
                    relation = relation.strip()
                    try:
                        target_str, relation_type = relation.split(' ', 1)
                        target_id = get_region_id(target_str)
                        yield (source_id, target_id, relation_type)
                    except ValueError as e:
                        # print('Failed to parse relation: {}'.format(relation))
                        continue
            else:
                target_str, relation_type = parts[1].strip().split(' ', 1)
                target_id = get_region_id(target_str)
                yield (source_id, target_id, relation_type)
            
        triplets = []
        for rel in relations:
            if rel.count(':') == 1:
                for triplet in get_triplet(rel):
                    triplets.append(triplet)
        return triplets
    
class SGEval(OspreyEval, MultiRegionDataset, SceneGraphParser):
    ''' 
    Scene Graph evaluation class that assigns generation result to scene graph object and relationship class
    using BertModel.
    '''

    def __init__(
        self, 
        model_path, 
        max_regions=150,
        bert_model=None,
        category=None,
        region_mode: Literal['box', 'segmentation', 'box_segmentation']='segmentation',
        use_object_description_context: bool=True,
        chunk_idx:int=0,
        num_chunks:int=1,
        debug=False
    ):
        
        super().__init__(model_path, chunk_idx=chunk_idx, num_chunks=num_chunks, debug=debug)
        self.max_regions = max_regions

        self.region_mode = region_mode
        self.is_train = False
        self.use_object_description_context = use_object_description_context

        SceneGraphParser.__init__(self, bert_model=bert_model, category=category)
    
    def create_mask_input(
        self, 
        regions: list[dict], 
        height: int,
        width: int,
        sort_regions_by_largest: bool = True, 
        top_k=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create mask input (torch.Tensor) for the given regions.
        
        Optionally can be sorted by largest size and limited to a certain number of regions.
        
        Args:
            regions (list[dict]): A list of dictionaries representing the regions, typically from sam predictions
            height (int): The height of the image.
            width (int): The width of the image.
            sort_regions_by_largest (bool, optional): Whether to sort the regions by largest size. Defaults to True.
            top_k (int, optional): The maximum number of regions to include. Defaults to None, which gets all regions.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The bounding boxes, segmentations, and input masks for the regions.

        """

        # def detections_to_regions(detections: sv.Detections) -> list[dict]:
        #     return [{'bbox': d.xyxy, 'segmentation': d.mask, 'area': d.area} for d in detections]
            
        # if isinstance(regions, sv.Detections):
        #     regions = detections_to_regions(regions)
        
        if sort_regions_by_largest:
            regions = self.get_topk_largest_regions(regions, top_k=top_k)
        else:
            regions = regions[:top_k]

        boxes, segs = self.process_regions(regions) # np.array([a['bbox'] for a in datum['regions']]) # xyxy

        masks: np.ndarray = self.create_masks(boxes, segs, height, width)

        return boxes, segs, masks
    
    # Prompt
    def generate_scene_graph(
        self, 
        image: str | Image.Image, 
        masks: torch.tensor, # [num_objects, H, W]
        bbox_texts=None,
        temperature=0.2, 
        top_p=1.0,
        max_new_tokens=1024
    ) -> tuple[list[str], list[str]]:
        """
        Generates scene graph for given regions.
        """
        # Generate scene graph for given regions
        image: Image.Image = self.load_image(image)
        num_objects = len(masks)

        # Generate object and relations relations for each region.
        object_outputs: list[str] = []
        relation_outputs: list[str] = []
        for id in range(num_objects):
            
            # Get object labels
            object_output: str = self.generate_object_description(image, masks, id, bbox_texts, temperature, top_p, max_new_tokens)  
            object_outputs.append(object_output)

            # Get Relations Per Object
            relation_output: str = self.generate_relation(image, masks, id, object_output, bbox_texts, temperature, top_p, max_new_tokens)
            relation_outputs.append(relation_output)

        return object_outputs, relation_outputs
    
    def generate_object_description(
        self, 
        image: Image.Image, 
        masks: torch.Tensor, 
        region_id: int, 
        bbox_texts=None,
        temperature=0.2, 
        top_p=1.0,
        max_new_tokens=1024
    ) -> str:
        """
        Describes objects in given regions for specified region_id.
        """

        num_objects = len(masks)
        assert region_id < num_objects
        w,h = image.size

        
        begin_string = self.get_region_string(n=num_objects, bbox_texts=bbox_texts)

        subj_region: str = self.textify_region(region_id)

        # Get object labels
        prompt: str = begin_string + ' ' + RELATION_DESCRIPTION_QUESTIONS[0]
        prompt = prompt.format(subj_region)
        init_inputs: dict = self.get_init_inputs(image,
                                    self.image_processor,
                                    masks=masks,
                                    prompt=prompt,
                                    )
        image = init_inputs['image']
        masks = init_inputs['masks'].cuda()

        conv = self.get_new_conv()
        qs = init_inputs['sources'][0][0]['value']
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        object_output: str = self.get_outputs(image, input_ids, masks, self.stop_str, 
                                        temperature, top_p, max_new_tokens=max_new_tokens)

        logging.debug(f"Prompt: {prompt}")
        logging.debug(f"Object Output: {object_output}")

        return object_output
    
    def generate_relation(self, 
        image: Image.Image, 
        masks: torch.Tensor, 
        region_id: int,
        object_output: str=None, 
        bbox_texts=None,
        temperature=0.2, 
        top_p=1.0, 
        max_new_tokens=1024
    ) -> str:
        """ Describes relations between objects in given regions for specified region_id. """

        num_objects = len(masks)
        assert region_id < num_objects

        begin_string = self.get_region_string(n=num_objects, bbox_texts=bbox_texts) 
        subj_region: str = self.textify_region(region_id)

        # Create prompt for relation generation
        relation_question = RELATION_QUESTIONS[0].format(subj_region)
        if self.use_object_description_context: # Use object description as context
            assert object_output is not None, "Object output is required for generating relations with object description context."
            prompt: str = begin_string + ' ' + RELATION_DESCRIPTION_QUESTIONS[0]
            prompt = prompt.format(subj_region)
            init_inputs: dict = self.get_init_inputs(image,
                                        self.image_processor,
                                        masks=masks,
                                        prompt=prompt,
                                        )
            image = init_inputs['image']
            masks = init_inputs['masks'].cuda()
            conv = self.get_new_conv()
            qs = init_inputs['sources'][0][0]['value']
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], object_output)
            conv.append_message(conv.roles[0], relation_question)
            conv.append_message(conv.roles[1], None)
        else: # Generate relations without object description context
            prompt: str = begin_string + ' ' + relation_question
            prompt = prompt.format(subj_region)
            init_inputs: dict = self.get_init_inputs(image,
                                    self.image_processor,
                                    masks=masks,
                                    prompt=prompt,
                                    )
            image = init_inputs['image']
            masks = init_inputs['masks'].cuda()
            conv = self.get_new_conv()
            qs = init_inputs['sources'][0][0]['value']
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        region_output: str = self.get_outputs(image, input_ids, masks, self.stop_str, 
                                        temperature, top_p, max_new_tokens=max_new_tokens)
        region_output = f"{subj_region}: {region_output}"

        logging.debug(f"Prompt: {prompt}")
        logging.debug(f"Object Output: {region_output}")

        return region_output
    
    def generate_holistic_scene_graph(self, image: str | Image.Image, masks: torch.Tensor, bbox_texts=None,
                                      temperature=0.2, top_p=1.0, max_new_tokens=1024) -> str:
        """
        Generates a holistic scene graph for all the regions.
        """
        image: Image.Image = self.load_image(image)
        num_objects = len(masks)
        begin_string = self.get_region_string(n=num_objects, bbox_texts=bbox_texts) 
        prompt: str = begin_string + ' ' + SG_QUESTIONS[0]
        init_inputs: dict = self.get_init_inputs(image,
                                    self.image_processor,
                                    masks=masks,
                                    prompt=prompt,
                                    )
        image = init_inputs['image']
        masks = init_inputs['masks'].cuda()

        conv = self.get_new_conv()
        qs = init_inputs['sources'][0][0]['value']
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        sg_outputs: str = self.get_outputs(image, input_ids, masks, self.stop_str, 
                                        temperature, top_p, max_new_tokens=max_new_tokens)
        
        logging.debug(f"Prompt: {prompt}")
        logging.debug(f"Object Output: {sg_outputs}")
        
        return sg_outputs
    

class PSGEval(SGEval):
    
    def create_segmentations(self, pan_seg_image, segments_info, ):
        img_bgr = cv2.imread(str(pan_seg_image))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = rgb2id(img_rgb)
        segmentations = []
        for segment_info in segments_info:
            mask = im == segment_info['id']
            segmentations.append(mask)
            assert np.sum(mask) > 0
            
        return np.array(segmentations, dtype=bool)