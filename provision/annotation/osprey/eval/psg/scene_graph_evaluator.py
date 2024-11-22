'''
Evaluation for SG generation
Measures recall and mean recall for subject, object and predicate.
'''
from abc import abstractmethod, ABC
import ast
import base64
import io
import json
import os
import re
import traceback
from typing import List, Dict
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import supervision as sv
import torch
from torchvision.ops import box_iou

import concurrent.futures
import multiprocessing as mp

from osprey.eval.openai_utils import OpenaiAPI
from osprey.eval.draw_utils import visualize_masks
# from osprey.verifier import Verifier

tqdm.pandas()

## Scene Graph Evaluation
class SceneGraphEvaluator(ABC):
    """ Factory class for Scene Graph Evaluation """
    registry = {}

    def __init__(self, name):
        self.name = name

    @classmethod
    def register(cls, name: str):
        """
        Class decorator to register a subclass with a given name.
        """
        def decorator(subclass):
            cls.registry[name] = subclass
            return subclass
        return decorator
    
    @classmethod
    def get_evaluator(cls, name: str, **kwargs) -> 'SceneGraphEvaluator':
        """
        Get an evaluator class by name from the registry.
        """
        if name not in cls.registry:
            raise ValueError(f"Unknown evaluator '{name}'. Available evaluators: {list(cls.registry.keys())}")
        return cls.registry[name](name, **kwargs)
    
    @classmethod
    def list_evaluators(cls):
        """
        List all evaluators in the registry.
        """
        return list(cls.registry.keys())
    
    @abstractmethod
    def __call__(self, sg: list[dict], **kwargs) -> Dict:
        raise NotImplementedError
    
    @staticmethod
    def get_single_bbox_iou(bbox1, bbox2) -> float:
        assert len(bbox1) == 4 and len(bbox2) == 4
        iou = box_iou(torch.tensor([bbox1]), torch.tensor([bbox2]))
        return iou.item()

@SceneGraphEvaluator.register('sg_recall')
class SceneGraphRecall(SceneGraphEvaluator):

    def __call__(self, data: list, category: dict[str, list | np.ndarray], **kwargs):
        ''' 
        Returns object recall, recall, mean recall for triplets.
        '''
        gt_lists = []
        pred_lists = []
        for row in tqdm(data):
            gt_list = []
            gt_boxes = row['gt_boxes']
            gt_object_labels = row['gt_object_labels']
            for label in row['gt_labels']:
                subj, obj, pred = label
                subj_label = gt_object_labels[subj]
                obj_label = gt_object_labels[obj]
                gt_list.append((subj_label, gt_boxes[subj], obj_label, gt_boxes[obj], pred))
            gt_lists.append(gt_list)
            
            pred_list = []
            pred_boxes = row['pred_boxes']
            pred_object_labels = row['pred_object_labels']
            for label in row['pred_labels']:
                subj, obj, pred = label
                try:
                    subj_label = pred_object_labels[subj]
                    obj_label = pred_object_labels[obj]
                    pred_list.append((subj_label, pred_boxes[subj], obj_label, pred_boxes[obj], pred))
                except IndexError:
                    pass
            pred_lists.append(pred_list)

        return {self.name: self._eval_sg(pred_lists, gt_lists, category)}

    def _eval_sg(self, pred_lists: list[list[tuple]], gt_lists: list[list[tuple]], category, plot_mean_recall=False):
        ''' Helper function to evaluate scene graph '''
        object_recall = []
        recall = []
        mean_recall = defaultdict(list)

        for preds,gts in tqdm(zip(pred_lists, gt_lists), total=len(pred_lists)):
            for gt in gts:
                object_match = False
                match = False
                gold_subj, gold_subj_bbox, gold_obj, gold_obj_bbox, gold_pred = gt
                for pred in preds:

                    pred_subj, pred_subj_bbox, pred_obj, pred_obj_bbox, pred_pred = pred

                    subj_iou = self.get_single_bbox_iou(pred_subj_bbox, gold_subj_bbox)
                    obj_iou = self.get_single_bbox_iou(pred_obj_bbox, gold_obj_bbox)

                    match = (
                        pred_subj == gold_subj and
                        subj_iou >= 0.5 and
                        pred_obj == gold_obj and
                        obj_iou >= 0.5 and
                        pred_pred == gold_pred
                    )

                    if pred_subj == gold_subj and subj_iou >= 0.5:
                        object_match = True

                    if match:
                        break
                object_recall.append(object_match)
                recall.append(match)
                mean_recall[gold_pred].append(match)

        mean_recall_list = []
        for k, v in mean_recall.items():
            k = category['predicate_classes'][k]
            mean_recall_list.append(sum(v) / len(v))
            print(f'Recall({k}): {sum(v) / len(v) * 100:.2f}')

        object_recall = sum(object_recall) / len(object_recall) * 100
        print(f'Object Recall: {object_recall:.2f}')

        recall = sum(recall) / len(recall) * 100
        print(f'Recall: {recall:.2f}')

        avg_mean_recall = sum(mean_recall_list) / len(mean_recall_list) * 100
        print(f'Mean Recall for {len(mean_recall_list)} predicates: {avg_mean_recall:.2f}')

        if plot_mean_recall:
            predicate_classes = category['predicate_classes']
            plt.figure(figsize=(10, 8))
            plt.bar(range(len(category['predicate_classes'])), mean_recall_list, color='skyblue')
            plt.xlabel('Predicate')
            plt.ylabel('Mean Recall')
            plt.xticks(range(len(predicate_classes)), predicate_classes, rotation=45, ha="right")
            plt.title('Mean Recall by Predicate')
            plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
            plt.show()

        return {
            'object_recall': object_recall,
            'recall': recall, 
            'mean_recall': avg_mean_recall, 
            'mean_recall_list': mean_recall_list
        }

@SceneGraphEvaluator.register('pred_recall')
class SceneGraphPredicateRecall(SceneGraphEvaluator):
    ## Predicate Classification Evaluation
    def __call__(self, data: list, category, topk_each_query=1000, **kwargs):
        '''
        Gather predictions by question_id
        '''
        id2graph = defaultdict(list)
        id2graph_gt = defaultdict(list)
        for datum in data:
            image_id = datum['image_id']

            # GT info
            relation = datum['relation']
            subj, obj, option = relation
            id2graph_gt[image_id].append((subj, obj, option))

            # Add Predicate Predictions 
            scores = datum['pred_scores']
            options = np.argsort(scores)[::-1]
            selected_options = options[:topk_each_query]
            for option in selected_options:
                score = scores[option]
                id2graph[image_id].append((subj, obj, option, score))

        for image_id in id2graph:
            id2graph[image_id] = sorted(id2graph[image_id], key=lambda x:x[-1], reverse=True) # sort by score
            id2graph[image_id] = [tuple(item[:-1]) for item in id2graph[image_id][:-1]] # remove score

        result = {}
        for topk in [20, 50, 100]:
            recall = []
            mean_recall = defaultdict(list)
            for image_id in id2graph_gt:
                graph = id2graph[image_id][:topk]
                graph_gt = id2graph_gt[image_id]
                for t in graph_gt:
                    recall.append(t in graph)
                    mean_recall[t[-1]].append(t in graph)

            mean_recall_list = []
            for k, v in mean_recall.items():
                k = category['predicate_classes'][k]
                mean_recall_list.append(sum(v) / len(v))
                print(f'Predicate Recall({k}): {sum(v) / len(v) * 100:.2f}')

            recall = sum(recall) / len(recall) * 100
            print(f'Recall@{topk}: {recall:.2f}')
            result[f'predicate_recall_{topk}'] = recall

            mean_recall = sum(mean_recall_list) / len(mean_recall_list) * 100
            print(f'Mean Recall@{topk} for {len(mean_recall_list)} predicates: {mean_recall:.2f}')
            result[f'mean_predicate_recall_{topk}'] = mean_recall
        
        return {self.name: result}
    
# Heuristics based evaluation
@SceneGraphEvaluator.register('model_rule')
class RuleEvaluator(SceneGraphEvaluator):

    def score_relations(self, image, subj, subj_bbox, obj, obj_bbox, pred, **kwargs) -> float:
        """ Score the relation between subject and object """
        raise NotImplementedError

    def __call__(self, data: list[dict], image, **kwargs) -> Dict:

        scores = []
        for datum in data:
            boxes = datum['pred_boxes']
            triplets = datum['pred_labels']
            for triplet in triplets:
                subj, obj, pred = triplet
                subj_bbox = boxes[subj] 
                obj_bbox = boxes[obj]

                score = self.score_relations(image, subj, subj_bbox, obj, obj_bbox, pred, **kwargs)
                scores.append(score)
            
        result = np.mean(scores)
        return {self.name: result}
   
# Model based evaluation 
class SceneGraphModelEvaluator(SceneGraphEvaluator):
    ''' Base class for model based evaluation '''
    system_prompt = None
    relation_question = "Is {subj} in {subj_bbox} {pred} {obj} in {obj_bbox}? Answer directly with 'yes' or 'no' only."
    include_image_region = True

    def create_prompt(self, subj: str, subj_bbox: list[int], obj:str, obj_bbox: list[int], pred: str, **kwargs) -> str:
        ''' Create prompt for the model '''
        return self.relation_question.format(subj=subj, subj_bbox=subj_bbox, obj=obj, obj_bbox=obj_bbox, pred=pred)
    
    @staticmethod
    def _draw_mask(image_rgb: np.ndarray, detection: sv.Detections):
        # Assuming visualize_masks is a function that draws a mask for a single detection
        image_mask_rgb = visualize_masks(image_rgb, detection, draw_bbox=True, draw_mask=True, label_text_position="center", white_padding=50, plot_image=False)
        image_mask = Image.fromarray(image_mask_rgb).convert('RGB')
        return image_mask
    
    def draw_masks(self, image_rgb: np.ndarray, detections_list: list[sv.Detections]) -> list[Image.Image]:
        ''' Draw masks for detections '''
        disable_tqdm = True
        n_processes = 16 # mp.cpu_count()
        with mp.Pool(processes=n_processes) as pool:
            results = list(tqdm(
                pool.starmap(
                    self._draw_mask, 
                    [(image_rgb, detection) for detection in detections_list]), 
                total=len(detections_list), disable=disable_tqdm
            ))
        return results

    @abstractmethod
    def generate_response(
        self, 
        image, 
        sg_inputs: list[dict], 
        **kwargs
    ) -> list[str]:
        ''' Returns model response for the given inputs '''
        raise NotImplementedError
    
    def __call__(
        self, 
        data: list[dict], 
        image_dir,
        return_predictions: bool = False,
        **kwargs
    ) -> Dict:

        def process_object_name(object_name: str) -> str:
            # strip punctuations and convert to lower case
            object_name = re.sub(r'[^\w\s]', '', object_name)
            object_name = object_name.lower()
            object_name = object_name.replace('"', '\\"')
            return object_name.strip()

        preds = []
        scores = []
        for idx, datum in enumerate(tqdm(data)):
            image_id = datum['image_id']
            boxes = datum['pred_boxes']

            triplets: list[tuple[int, int, str]] = datum['pred_triplets']
            object_names: list[str] = datum['pred_object_names']
            image_path = os.path.join(image_dir, image_id)
            sg_inputs = []
            for (subj_idx, obj_idx, pred) in triplets:
                subj = process_object_name(object_names[subj_idx])
                obj = process_object_name(object_names[obj_idx])
                subj_bbox = boxes[subj_idx] 
                obj_bbox = boxes[obj_idx]
                sg_inputs.append(dict(
                    subj=subj, 
                    subj_bbox=subj_bbox, 
                    obj=obj, 
                    obj_bbox=obj_bbox, 
                    pred=pred
                ))
            
            answers = self.generate_response(image_path, sg_inputs, verbose=idx==0, **kwargs)
            gt_scores = datum['scores']
            pred_scores = [self.get_score(answer) for answer in answers]
            
            for sg_input, answer, pred_score, gt_score in zip(sg_inputs, answers, pred_scores, gt_scores):
                correct = pred_score == gt_score
                scores.append(correct)
                pred = {
                    'image_id': image_id,
                    'answer': answer,
                    'pred_score': pred_score,
                    'gt_score': gt_score,
                    'correct': correct,
                }
                pred.update(sg_input)
                preds.append(pred)
            
        output = {'accuracy': np.mean(scores)}
        if return_predictions:
            output.update({'predictions': preds})
            
        return {self.name: output}
    
    def get_score(self, response: str) -> int:
        ''' Parse response from the model to get a binary score '''
        # use regex to strip any punctuations and non-alphanumeric characters
        response = re.sub(r'[^\w\s]', '', response)
        response = response.lower()

        # Check if 'yes' or 'no' is in the word list of the response
        if re.search(r'\byes\b', response):
            return 1
        elif re.search(r'\bno\b', response):
            return 0
        else:
            print(f"Unable to parse yes/no from response: {response}")
            return 0

class GPT4EvaluatorBase(SceneGraphModelEvaluator):
    """Base class for GPT4 evaluators."""
    image_detail = 'high'

    def __init__(
        self, 
        name, 
        gpt_model='gpt-4o-2024-08-06',
        max_threads=16,
        max_tries=3,
    ):
        self.name = name
        self.gpt_model = gpt_model
        self.max_threads = max_threads
        self.max_tries = max_tries

    def _generate_gpt_response(self, idx: int, prompt: str, image_input: list[Image.Image]):
        """ Helper function to generate GPT response for a single prompt. """
        openai_api = OpenaiAPI()
        
        if self.include_image_region:
            assert len(image_input) == 2
        c = 0
        while c < self.max_tries:
            try:
                response = openai_api.call_chatgpt_vision(
                    model=self.gpt_model,
                    sys_prompt=self.system_prompt,
                    usr_prompt=prompt,
                    image_input=image_input,
                    image_detail=self.image_detail,
                    temperature=0.0,
                )
                if response is None:
                    print(f"Failed to get response for prompt: {prompt}. Retrying...")
                    continue
                response, usage = response
                return idx, response
            except concurrent.futures.TimeoutError:
                print(f"TimeoutError for prompt: {prompt}. Retrying...")
            except Exception as e:
                traceback.print_exc()
                print(f"Exception for prompt: {prompt} - {e}. Returing 'None'")
                return idx, "None"
            c += 1
        print(f"Failed to get response for prompt: {prompt}. Retried {c} times. Returing 'None'")
        return idx, "None"
    
    def _generate_gpt_responses(self, prompts: list[str], image_inputs: list[list[Image.Image]]) -> list[str]:
        """Generate responses using multi-processing."""
        disable_tqdm = True
        n = min(self.max_threads, len(prompts))

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
            future_to_idx = {
                executor.submit(self._generate_gpt_response, idx, prompt, image_input): idx
                for idx, (prompt, image_input) in enumerate(zip(prompts, image_inputs))
            }
            
            results = []
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(prompts), disable=disable_tqdm):
                idx = future_to_idx[future]
                try:
                    result = future.result(timeout=10)
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    print(f"TimeoutError for prompt at index: {idx}")
                    results.append((idx, "None"))
                except Exception as e:
                    print(f"Exception for prompt at index: {idx} - {e}")
                    results.append((idx, "None"))

        # Sort results by the original index
        results.sort(key=lambda x: x[0])
        return [result[1] for result in results]
    
    def generate_response(self, image_path, sg_inputs: list[dict], **kwargs) -> list[str]:
        """Generate responses for inputs with bounding boxes."""

        prompts = []
        detections_list: list[sv.Detections] = []
        image = Image.open(image_path)
        verbose = kwargs.get('verbose', False)
        w,h = image.size

        for relation in sg_inputs:
            subj = relation['subj']
            subj_bbox = np.array(relation['subj_bbox'], dtype=int)
            obj = relation['obj']
            obj_bbox = np.array(relation['obj_bbox'], dtype=int)
            pred = relation['pred']

            # Create prompt
            prompt = self.relation_question.format(
                width=w,
                height=h,
                subj=subj,
                subj_bbox=subj_bbox,
                obj=obj,
                obj_bbox=obj_bbox,
                pred=pred
            )
            prompts.append(prompt)

            detections = sv.Detections(np.array([subj_bbox, obj_bbox]))
            detections_list.append(detections)
        
        # Prepare image inputs as lists of [image, image_mask]
        if self.include_image_region:
            image_rgb = np.asarray(image)
            image_masks: list[Image.Image] = self.draw_masks(image_rgb, detections_list)
            image_inputs = [[image_path, image_mask] for image_mask in image_masks]
        else:
            image_inputs = [image_path] * len(sg_inputs)

        # Generate GPT responses
        gpt_responses = self._generate_gpt_responses(prompts, image_inputs)

        if verbose:
            print("System Prompt:", self.system_prompt)
            print("Prompt:", prompts[0])
            print("Answer:", gpt_responses[0])
            print("Image:", image_inputs[0])

        return gpt_responses


@SceneGraphEvaluator.register('model_gpt4')
class GPT4Evaluator(GPT4EvaluatorBase):
    system_prompt = """
Your job is to verify if the relationship between the two objects holds true.
Consider the spatial position, size, and shape of the objects in the image to determine if the relationship is correct.
The first image shows the entire scene.
The second image shows the image and highlighted regions of interest for the two objects, enclosed in bounding boxes labeled as [0] and [1].
Object in region[0] is enclosed in purple bounding box and object in reion[1] is enclosed in red bounding box.
Bounding boxes are provided for the two objects in xyxy format: [x1,y1,x2,y2], where x1,y1 is the top-left corner and x2,y2 is the bottom-right corner of the bounding box.
Use this information to precisely locate the objects in the image.
Answer No if you believe one of the objects is not in the correct position or if the relationship is incorrect.
The statement is in form: '{obj0}' in region[0] {obj0_bbox} '{relationship}' '{obj1}' in region[1] {obj1_bbox}.
"""
    relation_question = "The size of this image is {width}x{height} (w,h). Is this statement correct: '{subj}' in region[0] {subj_bbox} '{pred}' '{obj}' in region[1] {obj_bbox}. Answer directly with 'yes' or 'no' only."
    include_image_region = True

@SceneGraphEvaluator.register('model_gpt4_no_bbox')
class GPT4EvaluatorNoBbox(GPT4EvaluatorBase):
    ''' GPT4 evaluator without bounding boxes in the prompt '''
    system_prompt = """
Your job is to verify if the relationship between the two objects holds true.
Consider the spatial position, size, and shape of the objects in the image to determine if the relationship is correct.
Answer No if you believe one of the objects is not in the correct position or if the relationship is incorrect.
The statement is in form: '{obj0}' '{relationship}' '{obj1}'.
"""
    relation_question = "Is this statement correct: '{subj}' '{pred}' '{obj}'. Answer directly with 'yes' or 'no' only."
    include_image_region = True


@SceneGraphEvaluator.register('model_gpt4_no_region')
class GPT4EvaluatorNoRegion(GPT4EvaluatorBase):
    ''' GPT4 evaluator without highlighted region in the image '''
    system_prompt = """
Your job is to verify if the relationship between the two objects holds true.
Consider the spatial position, size, and shape of the objects in the image to determine if the relationship is correct.
The first image shows the entire scene.
The second image shows the image and highlighted regions of interest for the two objects, enclosed in bounding boxes labeled as [0] and [1].
Object in region[0] is enclosed in purple bounding box and object in reion[1] is enclosed in red bounding box.
Use this information to precisely locate the two objects in the image.
Answer No if you believe one of the objects is not in the correct position or if the relationship is incorrect.
The statement is in form: '{obj0}' in {obj0_bbox} '{relationship}' '{obj1}' in {obj1_bbox}.
"""
    relation_question = "The size of this image is {width}x{height} (w,h). Is this statement correct: '{subj}' in {subj_bbox} '{pred}' '{obj}' in  {obj_bbox}. Answer directly with 'yes' or 'no' only."
    include_image_region = True

@SceneGraphEvaluator.register('model_gpt4_no_bbox_and_region')
class GPT4EvaluatorNoBboxRegion(GPT4EvaluatorBase):
    ''' GPT4 evaluator without bounding boxes and highlighted region in the image '''
    system_prompt = """
Your job is to verify if the relationship between the two objects holds true.
Consider the spatial position, size, and shape of the objects in the image to determine if the relationship is correct.
Answer No if you believe one of the objects is not in the correct position or if the relationship is incorrect.
The statement is in form: '{obj0}' '{relationship}' '{obj1}'.
"""
    relation_question = "Is this statement correct: '{subj}' '{pred}' '{obj}'. Answer directly with 'yes' or 'no' only."
    include_image_region = False


class Qwen2VLEvaluatorBase(SceneGraphModelEvaluator):
    """Base class for Qwen2VL evaluators."""
    def __init__(
        self,
        name: str,
        model_path='Qwen/Qwen2-VL-7B-Instruct',
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=512,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        verbose: bool = True,
    ):
        super().__init__(name=name)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.verbose = verbose

        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

        assert model_path is not None
        self.model_path = model_path
        self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        if model_path in ['Qwen2-VL-72B-Instruct', 'Qwen2-VL-72B-Instruct-GPTQ-Int8']:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
            ).eval()
            torch.cuda.empty_cache()
        else:
            device = torch.cuda.current_device()
            self.device = device
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,).eval()
            self.model = self.model.to(device)
            torch.cuda.empty_cache()
        
        
    @staticmethod
    def process_bbox(bbox: list[float | int], height: int, width: int) -> list[float]:
        """ Converts bbox from 0 to 1 to absolute values. """
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = x1/width, y1/height, x2/width, y2/height
        bbox = np.array([x1, y1, x2, y2])*1000
        bbox = np.array(bbox, dtype=int)
        return f"({bbox[0]},{bbox[1]}),({bbox[2]},{bbox[3]})"
    
    def generate_response(self, image: str | Image.Image, sg_inputs: list[dict], **kwargs) -> list[str]:
        """Generate responses for inputs with bounding boxes."""

        prompts = []
        detections_list: list[sv.Detections] = []
        if isinstance(image, str):
            image = Image.open(image)
        verbose = kwargs.get('verbose', False)
        w,h = image.size

        for relation in sg_inputs:
            subj = relation['subj']
            subj_bbox = relation['subj_bbox']
            obj = relation['obj']
            obj_bbox = relation['obj_bbox']
            pred = relation['pred']

            # Create prompt
            prompt = self.relation_question.format(
                width=w,
                height=h,
                subj=subj,
                subj_bbox=self.process_bbox(relation['subj_bbox'], h, w),
                obj=obj,
                obj_bbox=self.process_bbox(relation['obj_bbox'], h, w),
                pred=pred
            )
            prompts.append(prompt)

            detections = sv.Detections(np.array([subj_bbox, obj_bbox]))
            detections_list.append(detections)
        
        # Prepare image inputs as lists of [image, image_mask]
        if self.include_image_region:
            image_rgb = np.asarray(image)
            image_masks: list[Image.Image] = self.draw_masks(image_rgb, detections_list)
            image_inputs = [[image, image_mask] for image_mask in image_masks]
        else:
            image_inputs = [[image]] * len(sg_inputs)

        # Generate GPT responses
        responses = self._generate_responses(prompts, image_inputs)

        if verbose:
            print("System Prompt:", self.system_prompt)
            print("Prompt:", prompts[0])
            print("Answer:", responses[0])
            print("Image:", image_inputs[0])

        return responses
    
    def encode_image(self, image: str | Image.Image):
        """Encode image input."""
        if isinstance(image, str):
            if os.path.isfile(image):
                return f"file://{image}"
            else:
                return image
        elif isinstance(image, Image.Image):
            return image
        else:
            raise ValueError(f"Invalid image input: {image}")
    
    def _generate_responses(self, prompts: list[str], image_inputs: list[list[Image.Image]]):
        def _generate_single_response(prompt: str, image_input: list[str |Image.Image]):
            """ Helper function to generate response for a single prompt. """
            
            from qwen_vl_utils import process_vision_info

            messages = []
            if self.system_prompt is not None:
                messages.append({'role': 'system', 'content': self.system_prompt})
            image_content = []
            for image in image_input:
                image = self.encode_image(image)
                image_content.append({
                    "type": "image", "image": image,
                })
            messages = [
                {
                    "role": "user",
                    "content": image_content + \
                    [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info([messages])
            inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
            inputs = inputs.to('cuda')

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **self.generate_kwargs,
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            out = self.processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = out[0]
            return response

        return [_generate_single_response(prompt, image_input) for prompt, image_input in zip(prompts, image_inputs)]
    

@SceneGraphEvaluator.register('model_qwen2vl-7b')
class Qwen2VL7BEvaluator(Qwen2VLEvaluatorBase):
    system_prompt = """
    Your job is to verify if the relationship between the two objects holds true.
    Consider the spatial position, size, and shape of the objects in the image to determine if the relationship is correct.
    The first image shows the entire scene.
    The second image shows the image and highlighted regions of interest for the two objects, enclosed in bounding boxes labeled as [0] and [1].
    Object in region[0] is enclosed in purple bounding box and object in reion[1] is enclosed in red bounding box.
    Bounding boxes are provided for the two objects in xyxy format: [x1,y1,x2,y2], where x1,y1 is the top-left corner and x2,y2 is the bottom-right corner of the bounding box.
    Use this information to precisely locate the objects in the image.
    Answer No if you believe one of the objects is not in the correct position or if the relationship is incorrect.
    """
    relation_question = "Is this statement correct: '<|object_ref_start|>{subj}<|object_ref_end|> in region[0] <|box_start|>{subj_bbox}<|box_end|>' \
    '{pred}' '<|object_ref_start|>{obj}<|object_ref_end|> in region[1] <|box_start|>{obj_bbox}<|box_end|>'. Answer directly with 'yes' or 'no' only."
    include_image_region = True

@SceneGraphEvaluator.register('model_qwen2vl-7b_no_bbox')
class Qwen2VL7BEvaluatorNoBbox(Qwen2VLEvaluatorBase):
    system_prompt = """
    Your job is to verify if the relationship between the two objects holds true.
    Consider the spatial position, size, and shape of the objects in the image to determine if the relationship is correct.
    The first image shows the entire scene.
    The second image shows the image and highlighted regions of interest for the two objects, enclosed in bounding boxes labeled as [0] and [1].
    Object in region[0] is enclosed in purple bounding box and object in reion[1] is enclosed in red bounding box.
    Use this information to precisely locate the objects in the image.
    Answer No if you believe one of the objects is not in the correct position or if the relationship is incorrect.
    """
    relation_question = "Is this statement correct: '{subj}' '{pred}' '{obj}'. Answer directly with 'yes' or 'no' only."
    include_image_region = True

@SceneGraphEvaluator.register('model_qwen2vl-7b_no_region')
class Qwen2VL7BEvaluatorNoRegion(Qwen2VLEvaluatorBase):
    system_prompt = """
    Your job is to verify if the relationship between the two objects holds true.
    Consider the spatial position, size, and shape of the objects in the image to determine if the relationship is correct.
    Bounding boxes are provided for the two objects in xyxy format: [x1,y1,x2,y2], where x1,y1 is the top-left corner and x2,y2 is the bottom-right corner of the bounding box.
    Use this information to precisely locate the objects in the image.
    Answer No if you believe one of the objects is not in the correct position or if the relationship is incorrect.
    """
    relation_question = "Is this statement correct: '<|object_ref_start|>{subj}<|object_ref_end|> at <|box_start|>{subj_bbox}<|box_end|>' \
    '{pred}' '<|object_ref_start|>{obj}<|object_ref_end|> at <|box_start|>{obj_bbox}<|box_end|>'. Answer directly with 'yes' or 'no' only."
    include_image_region = False

@SceneGraphEvaluator.register('model_qwen2vl-7b_no_bbox_and_region')
class Qwen2VL7BEvaluatorNoBboxRegion(Qwen2VLEvaluatorBase):
    system_prompt = """
    Your job is to verify if the relationship between the two objects holds true.
    Consider the spatial position, size, and shape of the objects in the image to determine if the relationship is correct.
    Answer No if you believe one of the objects is not in the correct position or if the relationship is incorrect.
    """
    relation_question = "Is this statement correct: '{subj}' '{pred}' '{obj}'. Answer directly with 'yes' or 'no' only."
    include_image_region = False

@SceneGraphEvaluator.register('model_qwen2vl-72b')
class Qwen2VL7BEvaluator(Qwen2VLEvaluatorBase):
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, model_path='Qwen/Qwen2-VL-72B-Instruct', **kwargs)

if __name__ == '__main__':
    DATA_FILES={
        'human_eval_zixian': 
            ('/net/nfs.cirrascale/mosaic/jamesp/Osprey/data/relation_grounded/human_eval_zixian_interaction_600.jsonl',
                '/net/nfs.cirrascale/mosaic/jamesp/images/vg/VG_100K_all/'),
        'hico_det': 
            ('/net/nfs.cirrascale/mosaic/jamesp/data/hico_det/test_positive_negative_1000.jsonl',
                '/net/nfs.cirrascale/mosaic/jamesp/data/hico_det/images'),
        'openimages':
            ('/net/nfs.cirrascale/mosaic/jamesp/Osprey/data/openimages/test_v6_pos_neg_triplets.jsonl',
                '/net/nfs.cirrascale/prior/jamesp/data/openimages/test'),
    }
    parser = argparse.ArgumentParser(description='Evaluate scene graph generation')
    parser.add_argument('--evaluator', type=str, default='model_gpt4', help='Evaluator to use')
    parser.add_argument('--data', help='dataset name', required=True, choices=DATA_FILES.keys())
    parser.add_argument('--image_dir', type=str, help='Path to image directory')
    parser.add_argument('--category', help='path to category classes', default='/net/nfs.cirrascale/mosaic/jamesp/Osprey/osprey/eval/psg/psg_category.json')
    parser.add_argument('--n_samples', help='n_samples to evaluate', type=int, default=None)
    parser.add_argument('--output', type=str, help='Path to output file')
    args = parser.parse_args()

    '''
    Example usage:
    
    # Human Labeled 
    python osprey/eval/psg/scene_graph_evaluator.py \
        --data human_eval_zixian \
        --output /net/nfs.cirrascale/mosaic/jamesp/Osprey/osprey/eval/psg/results/human_eval_zixian_interaction_600/gpt4o_results.json
    
    # HICO-DET
    python osprey/eval/psg/scene_graph_evaluator.py \
        --data hico_det \
        --output /net/nfs.cirrascale/mosaic/jamesp/Osprey/osprey/eval/psg/results/hico_det_test_1000/gpt4o_results.json
    
    # Openimages
    python osprey/eval/psg/scene_graph_evaluator.py \
        --data openimages \
        --output /net/nfs.cirrascale/mosaic/jamesp/Osprey/osprey/eval/psg/results/openimages_test_v6/gpt4o_results.json
    
    # TODO: PSG
    python osprey/eval/psg/scene_graph_evaluator.py \
        --image_dir  /net/nfs.cirrascale/mosaic/jamesp/images/coco/ \
        --n_samples 10 \
        --output /net/nfs.cirrascale/mosaic/jamesp/Osprey/osprey/eval/psg/results/psg_eval_gpt4o_results.json
    '''

    assert args.data in DATA_FILES
    data_file, image_dir = DATA_FILES[args.data]
    df = pd.read_json(data_file, lines=True, dtype=False)
    data = df.to_dict(orient='records')
    data = data[:args.n_samples]
    for datum in data:
        datum['pred_boxes'] = datum['boxes']
        datum['pred_object_names'] = datum['object_names']
        datum['pred_triplets'] = datum['triplets']

    evaluator = SceneGraphEvaluator.get_evaluator(args.evaluator)
    result: dict = evaluator(
        data,
        image_dir=image_dir,
        return_predictions=True,
    )
    print('{} Accuracy: {}'.format(args.evaluator, result[args.evaluator]['accuracy']))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=4)