from typing import List, Dict, Literal, Tuple
import random
import copy
import numpy as np
import torch
from PIL import Image

from dataclasses import dataclass
from .stage2_data import CustomDataset
from .prompts import REF_WAY, REF_WAY_NUM

from osprey.train.train import preprocess, preprocess_multimodal


@dataclass
class MultiRegionInfo:
    img_path: str
    boxes: list
    segmentations: list
    region_mapping: dict
    convs: list

class MultiRegionDataset(CustomDataset):
    
    def __init__(self,
                tokenizer=None,
                data_args=None,
                ann_file=None,
                img_prefix=None,
                max_regions=30,
                max_gt_per_img=20,
                is_train=True,
                region_mode: Literal['box', 'segmentation', 'box_segmentation']='segmentation',
                use_bbox_text=False,
                sample=None,
    ):

        self.max_regions = max_regions
        self.is_train = is_train
        self.region_mode = region_mode
        self.use_box = region_mode == 'box'
        self.begin_str = """<image>\nThis provides an overview of the picture.\n"""
        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img, 
                         use_bbox_text=use_bbox_text, sample=sample)
    
    def load_annotations(self, ann_file):
        """
        Loads annotations from a given annotation file.

        Args:
            ann_file (str): Path to the annotation file.

        Returns:
            list of dict: A list of dictionaries, where each dictionary contains:
                - 'img_path' (str): The path to the image file.
                - 'boxes' (list of lists): The bounding boxes for objects in the image. Each bounding box is represented as a list of four integers [x, y, width, height].
                - 'segmentations' (list of lists): The segmentation masks for objects in the image. Each mask is represented as a list of coordinates [[x1, y1], [x2, y2], ...].
                - 'convs' (list of dicts): A list of dictionaries, each representing a conversation. Each conversation dictionary contains a 'value' key, which is a string representing the conversation.

        Example:
        [
            {
                'img_path': '/path/to/image.jpg',
                'boxes': [[10, 20, 50, 50], [70, 80, 100, 100]],
                'segmentations': [[[10, 20], [30, 40], [50, 60]], [[70, 80], [90, 100], [110, 120]]],
                'convs': [{'value': 'Hello, how are you?'}]
            },
            ...
        ]

        This method should be overridden in subclasses to load the annotations specific to the dataset.
        """
        raise NotImplementedError
    
    @staticmethod
    def process_regions(regions: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """ Convert detected regions into boxes and segmentations."""
        boxes: list[np.ndarray] = [region['xyxy'] for region in regions]
        segmentations: list[np.ndarray] = [region['segmentation'] for region in regions]
        assert len(segmentations) > 0
        assert len(boxes) == len(segmentations), \
            "number of boxes: ({}) and segmentations: ({}) should match".format(len(boxes), len(segmentations))
        return boxes, segmentations
   
    @staticmethod 
    def get_region_id(id: str, region_mapping: Dict[str, int]) -> int:
        """ Get the region id (in int) from the region mapping."""
        return int(region_mapping[id]) + 1  # (1-indexed)
    
    def get_region_string(self, n: int, bbox_texts: list[str]=None):
        """
        Generates a string of region references based on the number of regions and optional bbox texts.

        Args:
            n (int): The number of regions.
            bbox_texts (list[str], optional): A list of bbox texts, same size as `n`. Defaults to None.

        Returns:
            str: A string of region references.
        """
        if bbox_texts is not None:
            assert len(bbox_texts) == n
        
        ref_string = ''
        for i in range(n):
            if not bbox_texts:
                ref_string = ref_string +  f'region{i+1} <mask><pos>, '
            else:
                ref_string = ref_string +  f'region{i+1} <mask><pos> {bbox_texts[i]}, '
        ref_string = ref_string[:-2] # remove the last comma

        # ref_prefix = random.choice(REF_WAY)
        ref_prefix = random.choice(REF_WAY_NUM).format(n)
        region_string = ref_prefix.replace('<region>', ref_string)
        region_string = region_string

        return region_string
    
    def textify_region(self, region_id: int) -> str:
        ''' Returns a string representation of the region id.'''
        return 'region' + str(region_id+1)
    
    def create_masks(self, boxes: list, segmentations: list, h, w) -> np.ndarray:
        """
        Args:
            boxes (list): the ground truth bounding boxes
            segmentations (list): list of ground truth segmentation masks
            h (int): the height of the image
            w (int): the width of the image

        Returns:
            np.ndarray: returned segmentation mask, one for each object in the image.
        """

        # pred_masks = np.zeros((len(segmentations), h, w))
        # for i in range(len(pred_masks)):

        #     pred_mask = None
        #     bbox = boxes[i]
        #     mask = segmentations[i]
        #     pred_mask = self._create_single_mask(bbox, mask, h, w)
        #     pred_masks[i] = pred_mask
            
        # return pred_masks

        pred_masks = []
        for box, mask in zip(boxes, segmentations):
            pred_mask: np.ndarray = self._create_single_mask(box, mask, h, w)
            pred_masks.append(pred_mask)
        
        return np.array(pred_masks)

    
    def _create_single_mask(self, bbox: list | np.ndarray, mask, h, w) -> np.ndarray:
        """
        Creates a single mask based on the region mode.

        Args:
            bbox (np.ndarray): the bounding box for the object
            mask : the segmentation mask for the object. can be np array, or coco rle format.
            h (int): the height of the image
            w (int): the width of the image

        Returns:
            np.ndarray: the created mask for the object
        """

        pred_mask = None
        
        if self.region_mode == 'box':
            pred_mask = self.bboxToMask(bbox, h, w)
        elif 'segmentation' in self.region_mode:
            if self.region_mode == 'box_segmentation':
                pred_mask = None if mask is None else self.annToMask(mask, h, w)
                pred_mask = self.bboxToMaskWithBorder(bbox, h, w, border_thickness=3, internal_mask=pred_mask)
            elif self.region_mode == 'segmentation':
                pred_mask = self.annToMask(mask, h, w)
        
        if pred_mask is None:
            raise ValueError("Invalid region mode: %s" % self.region_mode)
        
        return pred_mask


    def __getitem__(self, i, debug=False):
        """
        Retrieves an item from the dataset at a given index.

        Args:
            i (int): The index of the item to retrieve.
            debug (bool, optional): If True, provides additional debug information for the conversation. Defaults to False.

        Returns:
            dict: A dictionary containing the processed image, its associated masks, and conversation tokens.

        The expected format of `data_infos` is a list of dictionaries, where each dictionary contains:
            - 'img_path': The path to the image file.
            - 'boxes': Absolute xyxy bounding boxes for objects in the image.
            - 'segmentations': The segmentation masks for objects in the image.
            - 'convs': A list of dictionaries, each representing a conversation.
        """
        data_info = self.data_infos[i]

        img_path = data_info['img_path']
        boxes = data_info['boxes']
        segmentations = data_info['segmentations']
        convs: List[Dict] = data_info['convs']

        image, image_size, image_token_len = self.process_image(img_path)
        w, h = image_size

        # Load masks as bbox, segm, or bbox + segm
        pred_masks: np.ndarray = self.create_masks(boxes, segmentations, h, w)

        convs = copy.deepcopy(convs)

        # Add image and region prefix
        num_objects = len(pred_masks)
        if self.use_bbox_text:
            if boxes is None:
                bbox_texts = [self.mask_to_bbox_text(mask) for mask in pred_masks]
            else:
                bbox_texts = [self.bbox_to_text(bbox, h, w) for bbox in boxes]
            region_string = self.get_region_string(num_objects, bbox_texts)
        else:
            region_string = self.get_region_string(num_objects)
        convs[0]['value'] = self.begin_str + region_string + convs[0]['value']

        sources = preprocess_multimodal(
            copy.deepcopy([convs]),
            self.data_args, 
            image_token_len
        )
        
        if debug:
            for conv in sources[0]:
                print(conv['from'])
                print(conv['value'])
                print("=")

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict['input_ids'][0],
                             labels=data_dict['labels'][0])

        data_dict['image'] = image
        data_dict['masks'] = torch.Tensor(pred_masks)

        return data_dict
    
    # def draw_segmentation(self, idx: int, image_file):
    #     import supervision as sv
    #     info =  self.data_infos[idx]
    #     img_path = info['img_path']
    #     im = cv2.imread(img_path)
    #     boxes = np.array(info['boxes'])

    #     data = self.__getitem__(idx, debug=True)
    #     mask = np.array(data['masks'].numpy(), dtype=bool)
    #     ids = np.array(range(len(mask)))
    #     labels = [f"[{idx+1}]" for idx in ids]
    #     detections = sv.Detections(xyxy=boxes, mask=mask, class_id=ids)
    #     box_annotator = sv.BoxAnnotator()
    #     label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    #     annotator = sv.MaskAnnotator()
    #     annotated_image = im.copy()
    #     # annotated_image = box_annotator.annotate(scene=im.copy(), detections=detections, labels=labels)
    #     annotated_image = annotator.annotate(scene=annotated_image, detections=detections)
    #     annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    #     cv2.imwrite('gqa_cot_sg.jpg',image_file)
    #     return annotated_image