import argparse
import json
from pathlib import Path
from PIL import Image

from functools import partial
import os
import numpy as np
import cv2
import supervision as sv
import logging
import torch

from transformers import AutoTokenizer, CLIPImageProcessor
from osprey.utils import disable_torch_init
from osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM
from osprey.mm_utils import tokenizer_image_token
from osprey.conversation import conv_templates, get_stop_str
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.train.train import DataArguments

from osprey.eval.annotate.generate_regions import load_sam_mask_generator, load_sam2_mask_generator
from osprey.eval.eval import MultiRegionEval
from osprey.eval.psg.eval_psg import SceneGraphParser

# set logging format
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

SG_QUESTION = 'Generate scene graph for given regions.'
RELATION_DESCRIPTION_QUESTION = 'Generate a description for: {}.'
RELATION_QUESTION = 'Generate list of relationships for: {}.'

SPECIALIST_QUESTIONS = {
    'spatial': 'What are the spatial relationships for {}?',
    'functional': 'What are the functional relationships for {}?',
    'interactional': 'What are the interactional relationships for {}?',
    'social': 'What are the social relationships for {}?',
    'symbolic': 'What are the symbolic relationships for {}?',
    
}


def xywh2xyxy(xywh):
    x,y,w,h = xywh
    return [x, y, x+w, y+h]

def load_osprey_model(model_path, device='cuda'):
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=True
    )
    model = OspreyLlamaForCausalLM.from_pretrained(
                                            model_path,
                                            torch_dtype=torch.bfloat16,
                                            # attn_implementation='flash_attention_2',
                                            ).to(device)
    tokenizer.pad_token = tokenizer.unk_token

    
    spi_tokens = ['<mask>', '<pos>']
    tokenizer.add_tokens(spi_tokens, special_tokens=True)
    
    for m in model.modules():
        m.tokenizer = tokenizer

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(dtype=torch.float16, device=device)
    return model, tokenizer

class OspreySG(MultiRegionEval):
    def __init__(self, model, tokenizer, region_mode='segmentation', conv=None, max_regions=None):
        self.model = model
        self.model.eval()
        self.device = model.device

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.unk_token

        self.region_mode = region_mode
        self.max_regions = max_regions
    
        self.image_processor = CLIPImageProcessor(do_resize=True, size={"shortest_edge":512}, resample=3,  do_center_crop=False, crop_size={"height": 512, "width": 512},
                                                    do_rescale=True, rescale_factor=0.00392156862745098, do_normalize=True, image_mean=[0.48145466, 0.4578275, 0.40821073],
                                                    image_std=[0.26862954, 0.26130258, 0.27577711], do_convert_rgb=True, )
        if conv is None:
            # if 'mistral' in model_path:
            #     conv = 'mistral_instruct'
            # else:
            conv = 'osprey_v1'
        print('Using conversation template:', conv)
        self.conv = conv_templates[conv]
        self.stop_str = get_stop_str(self.conv)
        
        self.scene_graph_parser = SceneGraphParser()
    
    def detections_from_sam(self, regions: list[dict]) -> tuple[list, list]:
        boxes =  [xywh2xyxy(region['bbox']) for region in regions]
        segmentations = [self.annToMask(region['segmentation']) for region in regions]

        return boxes, segmentations
    
    def create_mask_input(
        self, 
        regions: list[dict], 
        height: int,
        width: int,
        sort_regions_by_largest: bool = True, 
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create mask input (torch.Tensor) for the given regions.
        
        Optionally can be sorted by largest size and limited to a certain number of regions.
        If self.max_regions is set, it will limit the number of regions to that value.
        
        Args:
            regions (list[dict]): A list of dictionaries representing the regions, typically from sam predictions
            height (int): The height of the image.
            width (int): The width of the image.
            sort_regions_by_largest (bool, optional): Whether to sort the regions by largest size. Defaults to True.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The bounding boxes, segmentations, and input masks for the regions.

        """
        
        if sort_regions_by_largest:
            regions = self.get_topk_largest_regions(regions, top_k=self.max_regions)
        else:
            regions = regions[:self.max_regions]

        boxes, segs = self.detections_from_sam(regions) # performs xywh bbox -> xyxy conversion    
        masks: np.ndarray = self.create_masks(boxes, segs, height, width)

        return regions, masks
    
    def generate_detailed_scene_graph(
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

        def create_scene_graph_text(objects: list[str], relations: list[str]) -> str:
            object_text = '\n'.join([f"region{idx}: {obj}" for idx,obj in enumerate(objects, 1)])
            relation_text = '\n'.join(relations)
            sg_text = f"Objects:\n{object_text}\n\nRelations:\n{relation_text}"
            sg_text = sg_text.encode('ascii', 'ignore').decode('ascii')
            return sg_text
        
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
        
        sg_text: str = create_scene_graph_text(object_outputs, relation_outputs)
        sg_result: dict = self._parse_scene_graph(sg_text)

        return sg_result, sg_text
    
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
        prompt: str = begin_string + ' ' + RELATION_DESCRIPTION_QUESTION
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
        relation_question = RELATION_QUESTION.format(subj_region)
        if object_output is not None: # Use object description as context
            prompt: str = begin_string + ' ' + RELATION_DESCRIPTION_QUESTION
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

    def generate_holistic_scene_graph(self, image: Image.Image, 
                                    masks: torch.Tensor, bbox_texts=None,
                                    temperature=0.5, top_p=0.95, max_new_tokens=1024
    ) -> tuple[dict, str] | str:
        """
        Generate a holistic, parsed scene graph for the given image and masks.
        Args:
            image (Image.Image): 
            masks (torch.Tensor): 
            bbox_texts (_type_, optional): bbox_text to use for each mask. Defaults to None.
            temperature (float, optional): Defaults to 0.5.
            top_p (float, optional): Defaults to 0.95.
            max_new_tokens (int, optional): Defaults to 1024.

        Returns:
            tuple[dict, str]: parsed scene graph with objects and relation triplets.
            str: raw scene graph output from the language model.
        """
        num_objects = len(masks)
        begin_string = self.get_region_string(n=num_objects, bbox_texts=bbox_texts) 
        prompt: str = begin_string + ' ' + SG_QUESTION
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
        sg_text: str = self.get_outputs(image, input_ids, masks, self.stop_str, 
                                        temperature, top_p, max_new_tokens=max_new_tokens)
        sg_text = sg_text.encode('ascii', 'ignore').decode('ascii')
        sg_result: dict = self._parse_scene_graph(sg_text)

        # parse scene graph output
        logging.debug(f"Prompt: {prompt}")
        
        return sg_result, sg_text    

    def _parse_scene_graph(self, scene_graph_output: str) -> dict:
        """
        Parses the scene graph output into object names and relation triplets.
        
        Args:
            scene_graph_output (str): raw scene graph output from the language model.

        Returns:
            dict: parsed scene graph output.
            {
                'objects': list[str],
                'relations': list[tuple[int, int, str]], # [subject_id, object_id, relation_name]
            }
        """
        parsed_sg = self.scene_graph_parser.parse_outputs(scene_graph_output, get_labels=False)

        if parsed_sg is None:
            return None
        return {
            'objects': parsed_sg['pred_object_names'],
            'relations': parsed_sg['pred_triplets'],
        }

# TODO: convert to huggingface pipeline
class SGRunner:
    def __init__(self, osprey_model: OspreySG, mask_generator=None):
        self.sg_model = osprey_model
        self.mask_generator = mask_generator
    
    def load_image(self, im):
        if isinstance(im, Image.Image):
            return im
        return Image.open(im).convert('RGB')

    def __call__(self, im, regions=None, sg_mode='whole'):

        im: Image.Image = self.load_image(im)
        width, height = im.size
        if regions is None: # generate regions if not provided
            regions: list[dict] = self.mask_generator.generate(np.asarray(im))
        if len(regions) == 0:
            logging.info(f"No regions found for image. Skipping scene graph generation.")
            return regions, None, None
        regions, masks = self.sg_model.create_mask_input(regions, height, width)
        masks = torch.from_numpy(masks)

        # generate scene graph
        if sg_mode == 'detailed':
            sg_result, sg_text = self.sg_model.generate_detailed_scene_graph(im, masks)
        else:
            sg_result, sg_text = self.sg_model.generate_holistic_scene_graph(im, masks)

        return regions, sg_result, sg_text

if __name__ == '__main__':
    '''
    python -m osprey.eval.annotate.generate_sg \
        --model exp/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_7b_stage3_v1_bs16/checkpoint-102315/ \
        --image_path /net/nfs.cirrascale/mosaic/jamesp/images/llava/LLaVA-Pretrain/images/00453/004539375.jpg \
        --output_file 004539375.pkl
       
    # sam-2?
    python -m osprey.eval.annotate.generate_sg \
        --model exp/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_7b_stage3_v1_bs16/checkpoint-102315/ \
        --mask_generator_path /net/nfs.cirrascale/mosaic/jamesp/models/segment-anything/sam2_hiera_large.pt \
        --mask_generator_model sam2 \
        --image_path /net/nfs.cirrascale/mosaic/jamesp/images/llava/LLaVA-Pretrain/images/00453/004539375.jpg \
        --output_file 004539375.pkl
         
    '''
    parser = argparse.ArgumentParser(description='osprey sg generator', formatter_class=argparse.RawTextHelpFormatter)
    # Model config
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--image_path', help='image to generate scene graph for', required=True) 
    parser.add_argument('--mask_generator_path', help='path to mask_generator', 
                        default='/net/nfs.cirrascale/mosaic/jamesp/models/segment-anything/sam_vit_h_4b8939.pth')
    parser.add_argument('--mask_generator_model', help='mask_generator model to use', default='sam', choices=['sam', 'sam2'])
    parser.add_argument('--sam_mode', help='custom mask generator mode to use for sam model', default='whole', choices=['whole', 'default', 'part'])

    parser.add_argument('--output_file', help='pickle file to save scene graph', default=None)
    parser.add_argument('--sg_mode', default='holistic', choices=['holistic', 'detailed'])

    # Region config
    parser.add_argument('--max_regions', type=int, default=99)
    parser.add_argument('--region_mode', type=str, default='segmentation', choices=['box', 'segmentation', 'box_segmentation'])

    # Gen config
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)

    args = parser.parse_args()

    # Load sg model
    model_path = args.model
    model, tokenizer = load_osprey_model(model_path, device='cuda')
    conv_mode = 'mistral_instruct' if 'mistral' in model_path else 'osprey_v1'
    osprey_sg = OspreySG(model, tokenizer, region_mode='segmentation', conv=conv_mode, max_regions=args.max_regions)

    # Load region proposal model
    if args.mask_generator_model == 'sam2':
        mask_generator = load_sam2_mask_generator(args.mask_generator_path, device='cuda')
    else:
        mask_generator = load_sam_mask_generator(args.mask_generator_path, sam_mode=args.sam_mode, device='cuda')

    image_path = args.image_path
    im = Image.open(args.image_path).convert('RGB')

    # generate scene graph with regions
    sg_runner = SGRunner(osprey_sg, mask_generator)
    regions, sg_result, sg_text = sg_runner(image_path, sg_mode=args.sg_mode)
    logging.info(f"Scene Graph Output: {sg_text}")
    logging.info(f"Parsed Scene Graph: {sg_result}")

    if args.output_file:
        import pickle
        from osprey.eval.draw_utils import visualize_masks
        
        mask_image: np.ndarray = visualize_masks(np.asarray(im), regions, plot_image=False)
        # mask_image = Image.fromarray(mask_image) # convert to PIL?

        # TODO: save the masks in `regions` as rle string instead of numpy array
        result = {
            'image_path': image_path,
            'mask_image': mask_image,
            'regions': regions,
            'sg_result': sg_result,
            'sg_text': sg_text,
        }
        with open(args.output_file, 'wb') as f:
            pickle.dump(result, f)
        print(f'Saved result to {args.output_file}')
        





    
    