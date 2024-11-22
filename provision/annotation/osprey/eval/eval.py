
from copy import deepcopy
import os
from typing import List, Union
from PIL import Image
import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader

import re
import wandb
from PIL import Image

from functools import partial
from transformers import AutoTokenizer, CLIPImageProcessor, AutoConfig
import supervision as sv
from pycocotools import mask as maskUtils

from osprey.train.train import DataArguments, preprocess_multimodal
from osprey.mm_utils import process_anyres_image, process_highres_image, process_highres_image_crop_split
from osprey.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, get_stop_str
from osprey.mm_utils import tokenizer_image_token
from osprey.utils import disable_torch_init
from osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM
from osprey.model.language_model.osprey_mistral import OspreyMistralForCausalLM
from osprey.model.multimodal_encoder.clip_encoder import  get_image_processor

from osprey.datasets.multi_region import MultiRegionDataset
from osprey.datasets.llava import get_whole_image_mask

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True


Ref_WAY = [
    'With <region> in the image, ',
    'Given <region>, ',
    'Given <region> in the image, ',
    'Several regions <region> are in the image and ',
]

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    return [lst[i::n] for i in range(n)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]
        
class OspreyEval:
    def __init__(self, model_path: str, device='cuda', chunk_idx=0, num_chunks=1, debug=False):

        disable_torch_init()
        model_path = os.path.expanduser(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="right",
            use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.unk_token
        self.image_processor = get_image_processor(img_size=512)

        # self.image_processor = CLIPImageProcessor(do_resize=True, size={"shortest_edge":512}, resample=3,  do_center_crop=True, crop_size={"height": 512, "width": 512},
        #                                           do_rescale=True, rescale_factor=0.00392156862745098, do_normalize=True, image_mean=[0.48145466, 0.4578275, 0.40821073],
        #                                           image_std=[0.26862954, 0.26130258, 0.27577711], do_convert_rgb=True, )
        
        spi_tokens = ['<mask>', '<pos>']
        self.tokenizer.add_tokens(spi_tokens, special_tokens=True)

        if debug:
            self.model = None
        else:
            config = AutoConfig.from_pretrained(model_path)
            if 'mistral' in config.model_type:
                self.model = OspreyMistralForCausalLM.from_pretrained(
                                                        model_path,
                                                        torch_dtype=torch.bfloat16,
                                                        ).to(device)
                conv = 'mistral_instruct'
            else:
                self.model = OspreyLlamaForCausalLM.from_pretrained(
                                                        model_path,
                                                        torch_dtype=torch.bfloat16,
                                                        ).to(device)
                conv = 'osprey_v1'
        
            for m in self.model.modules():
                m.tokenizer = self.tokenizer

            vision_tower = self.model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(dtype=torch.float16, device=device)
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.device = device
    
        print('Using conversation template:', conv)
        self.conv = conv_templates[conv]
        self.stop_str = get_stop_str(self.conv)
        # single_seps = [SeparatorStyle.SINGLE, SeparatorStyle.MPT, SeparatorStyle.PLAIN]

        # chunking data
        self.chunk_idx = chunk_idx
        self.num_chunks = num_chunks
    
    def get_new_conv(self):
        return self.conv.copy()
    
    def get_chunk(self, lst: list | dict) -> list[dict]:
        """
        Get a chunk of a list or dictionary.
        Used for eval with sharding.

        This method takes a list or dictionary and returns a chunk of it based on the number of chunks and the chunk index.

        Args:
            lst (list or dict): The data to get a chunk from.

        Returns:
            list: A chunk of the input list or dictionary.

        """

        if isinstance(lst, dict):
            lst = list(lst.items())
        
        return get_chunk(lst, self.num_chunks, self.chunk_idx)
    
    def get_region_prompt(self, n: int) -> str:
        """
        Generate prompt to describe regions based on number of objects.

        Args:
            n (int): number of objects.

        Returns:
            str: prompt for regions
        """
        ref_string = ''
        for i in range(n):
            ref_string = ref_string +  f'region{i+1} <mask><pos>' + ','
        ref_string = ref_string[:-1]
        ref_prefix = Ref_WAY[0]

        begin_string = ref_prefix.replace('<region>', ref_string)
        begin_string = begin_string + f'there are {n} part regions in the image.'

        return begin_string

    def load_image(self, image: Union[str, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert('RGB')
        return Image.open(image).convert('RGB')
    
    def generate(self, img_path: str, masks: torch.Tensor, prompt: Union[str, List[str]], 
                temperature=1.0, top_p=1.0, max_new_tokens=1024, **kwargs
    ) -> str:
        """Generate a text output based on an image, mask, and prompt.

        Args:
            img_path (str): The path to the image.
            masks (torch.Tensor): The masks for the image.
            prompt (Union[str, List[str]]): The prompt for generating the text output. A list should contain alternating sources and outputs.
            temperature (float, optional): The temperature for controlling the randomness of the output. Defaults to 1.0.
            top_p (float, optional): The top-p value for controlling the diversity of the output. Defaults to 1.0.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 1024.

        Returns:
            str: The generated text output.
        """

        if isinstance(prompt, list):
            assert len(prompt) % 2 == 1, 'Prompt should be a list of alternating sources and outputs'
        else:
            prompt = [prompt]
        init_inputs: dict = self.get_init_inputs(img_path,
                                            self.image_processor,
                                            masks=masks,
                                            prompt=prompt[0],
                                            )
        image = init_inputs['image']
        masks = init_inputs['masks'].cuda()

        conv = self.get_new_conv()
        qs = init_inputs['sources'][0][0]['value'] # Initial prompt with image tokens
        conv_prompt = [qs] + prompt[1:] +  [None]
        for idx, p in enumerate(conv_prompt):
            role = conv.roles[idx % 2]  # Alternate between user and model roles
            conv.append_message(role, p)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        outputs: str = self.get_outputs(image, input_ids, masks, self.stop_str, 
                                        temperature, top_p, max_new_tokens=max_new_tokens, **kwargs)

        return outputs
    
    def get_init_inputs(self,
                        img_path: Union[str, Image.Image],
                        processor: CLIPImageProcessor,
                        prompt: str,
                        masks: torch.Tensor,
            ):
        """
        Get the initial inputs for evaluation.

        Args:
            img_path (Union[str, Image.Image]): The path to the image or the image object itself.
            processor (CLIPImageProcessor): The image processor object.
            prompt (str): The prompt for the evaluation.
            masks (torch.Tensor): The masks for the image.

        Returns:
            dict: A dictionary containing the processed inputs for evaluation.
        """

        def get_image_token_len(image):
            patch_size: int = 16 # Default: 16
            image_token_len = 0
            for im in image:
                image_token_len += (im.shape[0] // patch_size) * (im.shape[1] // patch_size)
            return image_token_len
           
        image: Image.Image = self.load_image(img_path)

        image_aspect_ratio = self.model.config.image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt", do_center_crop=False)["pixel_values"][0].unsqueeze(0)
        else:
            image = processor.preprocess(image, return_tensors="pt", do_center_crop=False)["pixel_values"][0]
            image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                size=(512, 512),
                                                mode='bilinear',
                                                align_corners=False)

        image_token_len = get_image_token_len(image)

        if masks is not None:
            masks = masks.to(image.device)

        begin_str = """<image>\nThis provides an overview of the picture.\n"""

        sources = dict()
        sources['conversations'] = []
        sources['conversations'].append({'from': 'human', 'value': begin_str+prompt})
        
        sources = preprocess_multimodal([sources['conversations']], data_args, image_token_len)

        data_dict = {}
        data_dict['sources'] = sources
        data_dict['image'] = image
        data_dict['masks'] = masks
        return data_dict
    
    def get_outputs(
        self, 
        image: torch.Tensor, 
        input_ids: torch.Tensor, 
        masks: torch.Tensor, 
        # past_key_values=None,
        stop_str: str = None, 
        temperature=1.0, 
        top_p=1.0, 
        max_new_tokens=512, 
        do_sample=True,
        **kwargs
    ) -> str:
        """
        Generate outputs based on the given input.

        Args:
            image (torch.Tensor): The input image.
            input_ids (torch.Tensor): The input token IDs.
            masks (torch.Tensor): The segmentation masks.
            stop_str (str): The stop string to indicate the end of generation.
            temperature (float, optional): The temperature for sampling. Defaults to 1.0.
            top_p (float, optional): The top-p value for nucleus sampling. Defaults to 1.0.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.
            do_sample (bool, optional): Whether to use sampling or greedy decoding. Defaults to True.

        Returns:
            str: The generated outputs.
        """
        if stop_str is None:
            stop_str = get_stop_str(self.conv)
            
        self.model.model.tokenizer = self.tokenizer

        with torch.inference_mode():
            self.model.orig_forward = self.model.forward
            self.model.forward = partial(
                self.model.orig_forward,
                img_metas=[None],
                masks=[masks.half() if masks is not None else None]
            )
            output = self.model.generate(
                input_ids,
                images=image.unsqueeze(0).half().cuda(),
                # past_key_values=past_key_values,
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                use_cache=True,
                num_beams=1,
                **kwargs
            )

            self.model.forward = self.model.orig_forward

        output_ids = output

        # Parse output response
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:],
                                            skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs: str = outputs.strip()

        return outputs
    
    def _get_batch_scores(self, image_batch, mask_batch, input_ids_batch, labels_batch):
        """
        Computes the loss scores for a single batch of data.
        Useful for multiple choice evaluation based on perplexity.

        This helper function processes a batch of inputs with the model to obtain logits, computes the cross-entropy loss,
        and then calculates normalized and total loss scores for the batch.

        Args:
            input_ids_batch (torch.Tensor): The batch of tokenized input IDs.
            labels_batch (torch.Tensor): The batch of labels corresponding to the input IDs.
            image_batch (torch.Tensor): The batch of images corresponding to the input IDs.
            mask_batch (list[torch.Tensor]): The batch of masks for image regions.

        Returns:
            tuple: A tuple containing two lists, the first being the normalized scores and the second the total scores for the batch.
        """
        with torch.inference_mode():
            self.model.orig_forward = self.model.forward
            self.model.forward = partial(self.model.orig_forward, img_metas=[None] * len(mask_batch), masks=mask_batch)

            attention_mask = input_ids_batch.ne(self.tokenizer.pad_token_id)
            output = self.model(
                input_ids_batch,
                attention_mask=attention_mask,
                images=image_batch.half().cuda(),
            )
            self.model.forward = self.model.orig_forward

            shift_logits = output.logits[..., -labels_batch.size(1):-1, :].contiguous()
            shift_labels = labels_batch[..., 1:].contiguous()
            
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)  # Ensure labels are on the same device as logits

            loss = torch.nn.functional.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=IGNORE_INDEX,  # Assuming IGNORE_INDEX is defined elsewhere
                reduction='none'
            )

            loss = loss.view(input_ids_batch.shape[0], -1)

            norm_loss = loss.mean(-1) / (loss != 0).sum(dim=1)
            norm_scores = -norm_loss.detach().cpu().float().numpy()  # B

            scores = -loss.sum(-1).detach().cpu().float().numpy()  # B

        return norm_scores.tolist(), scores.tolist()

    def get_scores(self, image_batch: torch.Tensor , mask_batch: List[torch.Tensor], input_ids: torch.Tensor, labels: torch.Tensor, batch_size=24) -> dict:
        """
        Processes all input data in batches and computes loss scores.

        This function handles the batching of input data and uses the _get_batch_scores helper function to compute
        scores for each batch. It aggregates these scores across all batches to provide a comprehensive score result.

        Args:
            image_batch (torch.Tensor): The entire dataset of images to be processed.
            mask_batch (list[torch.Tensor]): The entire dataset of masks corresponding to the image regions.
            input_ids (torch.Tensor): Tokenized input IDs for all data points.
            labels (torch.Tensor): Labels corresponding to each set of input IDs.
            batch_size (int, optional): The size of each batch to be processed. Defaults to 32.

        Returns:
            dict: A dictionary containing 'norm_scores' and 'scores' lists with the normalized and total scores for all batches.
        """
   
        total_batches = (input_ids.size(0) + batch_size - 1) // batch_size  # Calculate total number of batches
        all_norm_scores = []
        all_scores = []

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, input_ids.size(0))

            input_ids_batch = input_ids[start_idx:end_idx].contiguous()
            labels_batch = labels[start_idx:end_idx].contiguous()
            image_batch_current = image_batch[start_idx:end_idx].contiguous()
            mask_batch_current = mask_batch[start_idx:end_idx]

            norm_scores, scores = self._get_batch_scores(image_batch_current, mask_batch_current, input_ids_batch, labels_batch)
            all_norm_scores.extend(norm_scores)
            all_scores.extend(scores)

        return {'norm_scores': all_norm_scores, 'scores': all_scores}
    
    def annToMask(self, mask_ann, h=None, w=None):
        if isinstance(mask_ann, np.ndarray):
            return mask_ann
        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, h, w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list): # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, h, w)
        else: # rle
            rle = mask_ann
        mask = maskUtils.decode(rle).astype(bool)
        return mask
    
    def get_topk_largest_regions(self, regions: list[dict], top_k: int) -> list[dict]:
        """ Get top k largest regions. """
        return sorted([r for r in regions], key=lambda x: x['area'], reverse=True)[:top_k]

    def get_topk_largest_regions_from_detections(self, detections: sv.Detections, top_k: int) -> sv.Detections:
        """ Get top k largest regions from detections. """
        areas = detections.area
        sorted_indices = sorted(range(len(areas)), key=lambda idx: areas[idx], reverse=True)
        return detections[sorted_indices[:top_k]]

    @staticmethod
    def get_region_id(region_str: str) -> int:
        """
        Extracts and converts the region ID from a string to 0-indexed integer.
        """
        match = re.search(r'region(\d+)', region_str)
        if match:
            return int(match.group(1)) - 1
        return -1

    
    @staticmethod
    def log_wandb(run_name: str, key: str, wandb_data, config=None):
        ''' Upload eval result as wandb artifact. '''
        print('Logging {}: {} to wandb run: {}'.format(key, wandb_data, run_name))
        import wandb
        api = wandb.Api()

        wandb_runs = list(api.runs(filters={"display_name": run_name}))
        try:
            print('Found exisitng wandb run: {}'.format(run_name))
            run = wandb_runs[0] # first run is the latest one by default 
            run_id = run.id
            wandb.init(
                name=run_name,
                resume='must',
                id=run_id
            )
        except IndexError:
            print('No existing wandb run found for {}'.format(run_name))
            print('Will create new run for {}'.format(run_name))
            wandb.init(
                run_name=run_name,
                tags=key,
                config=config,
                group='eval',
            )
        
        wandb.log(wandb_data, step=None)
        
        # Create artifact intead
        # artifact = wandb.Artifact(
        #     name=key,
        #     type='dataset'
        # )
        # artifact.new_file
        # wandb.log_artifact(artifact)

class MultiRegionEval(OspreyEval, MultiRegionDataset):
    ''' Class for evaluating models on multi-region datasets. '''
    def __init__(self, model_path, max_regions: int, region_mode: str, chunk_idx=0, num_chunks=1, debug=False):
        self.max_regions = max_regions
        self.region_mode = region_mode
        super().__init__(model_path, chunk_idx=chunk_idx, num_chunks=num_chunks, debug=debug)
    
class LLAVAEval(OspreyEval):
    ''' Class for evaluating models on LLAVA dataset with no regions but with only global image. '''
    def __init__(self, model_path, chunk_idx=0, num_chunks=1, debug=False):
        super().__init__(model_path, chunk_idx=chunk_idx, num_chunks=num_chunks, debug=debug)

        self.region_str =  "There is region1 <mask><pos> for the entire image. "
        self.question_str =  'Answer the question using a single word or phrase.'
    
    def get_single_mask(self, h, w):
        '''' Single global mask for the entire image. '''
        return np.array([get_whole_image_mask(h,w)])
    
    def get_question_prompt(self, question) -> str:
        # return self.region_str + self.question_str + '\n' + question
        return question + '\n' + self.question_str

        
