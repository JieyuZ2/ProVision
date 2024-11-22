'''
Scene graph parsing from generated scene graph.
'''
import ast
from functools import partial
import os
import glob
import math
import json
from pathlib import Path
import random
import pandas as pd
from typing import List, Tuple
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from osprey.eval.psg.eval_psg import SceneGraphParser
from osprey.eval.psg.scene_graph_evaluator import SceneGraphEvaluator
from osprey.eval.utils import load_hdf5_file, is_hdf5_file
    
def get_psg_gt(psg_file) -> dict[str, dict]:
    '''
    Get PSG ground truth data from PSG file
    - gt_labels [list[list[int, int, str]]]: [[object1, object2, relation], ...]
    - gt_object_labels [list[int]]: [object1, object2, ...]
    - gt_boxes: [[x1, y1, x2, y2], ...]
    '''
    gt_data = {}
    with open(psg_file) as f:
        psg_gt = json.load(f)

    for data in psg_gt['data']:
        id = str(data['index'])
        objects = data['segments_info']
        gt_object_labels = []
        gt_boxes = []
        for idx, obj in enumerate(objects):
            gt_object_labels.append(obj['category_id'])
            gt_boxes.append(data['annotations'][idx]['bbox'])
        
        gt_data[id] = {
            'image_id': data['file_name'],
            'labels': data['relations'],
            'object_labels': gt_object_labels,
            'boxes': gt_boxes,
        }
    
    return gt_data

def get_vg_gt(vg_file):
    with open(vg_file) as f:
        vg_gt = json.load(f)
    
    gt_data = {}
    for data in vg_gt['data']:

        objects = data['objects']
        object_labels = [obj['category_id'] for obj in objects]
        boxes = [obj['bbox'] for obj in objects] # xywh
        boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes] # xyxy
        gt_data[data['image_id']] = {
            'labels': data['relations'],
            'object_labels': object_labels,
            'boxes': boxes,   
        }
        
    return gt_data

def add_gt_data(df: pd.DataFrame, data_path: str) -> pd.DataFrame:
    if 'psg' in data_path:
        print('Loading PSG ground truth data..')
        gt_data: dict = get_psg_gt(data_path)
    else:
        print('Loading VG ground truth data..')
        gt_data: dict = get_vg_gt(data_path)
    for k in ['labels', 'object_labels', 'boxes']:
        df['gt_'+k] = df['id'].apply(lambda x: gt_data[str(x)][k])
    return df        

def process_gpt_sg(df: pd.DataFrame) -> pd.DataFrame:

    def create_scene_graph_raw_text(sg: dict):
        sg = {int(k): v for k, v in sg.items()}
        
        pred_boxes = []
        object_text = "Objects:\n"
        relation_text = "Relations:\n"
        for subj_id, obj in sorted(sg.items()):
            object_text += f"region{subj_id+1}: {obj['name']}\n"

            # Load relations as sg mode 0
            relation_obj_text = ""
            for rel_name, obj_ids in obj['rel'].items(): # {rel_name: [obj_ids]} 
                relation_obj_text += f"{rel_name}: {', '.join([f'region{obj_id+1}' for obj_id in obj_ids])};"
            if relation_obj_text:
                relation_text += f"region{subj_id+1}: {relation_obj_text}\n"
            
            # Get xyxy bbox
            pred_boxes.append(obj['bbox'])
        
        scene_graph_raw_text = f"{object_text}\n{relation_text}"
        return {
            'scene_graph_raw_text': scene_graph_raw_text,
            'pred_boxes': pred_boxes
        }
    
    res = df['updated_sg'].apply(create_scene_graph_raw_text)
    for k in ['scene_graph_raw_text', 'pred_boxes']:
        df[k] = res.apply(lambda x: x[k])
    
    return df

def save_metrics(output_file: str,  results: dict) -> None:
    output = {}
    # update existing output file with new results
    if os.path.isfile(output_file):
        with open(output_file, 'r') as f:
            output = json.load(f)
        print('Found existing metric from output file {}'.format(output_file))
    output.update(results)
    print(output)
    print('Saving result to.. {}'.format(output_file))
    with open(output_file, 'w') as f:
        json.dump(output, f)


def get_data(
        prediction: str | Path,
        data: str,
        sg_parser: SceneGraphParser, 
        sg_mode: int,
        top_k: int,
        n_samples: int = None
    ):
    """
    
    Returns dataframe with parsed scene graph data.
    Parsed scene graph should have the following columns:
    Pred:
        - pred_names: list of predicted relation names
        - pred_labels: list of predicted relation labels
        - pred_object_names: list of predicted object names
        - pred_object_labels: list of predicted object labels
        - pred_boxes: list of predicted bounding boxes
    GT:
        - gt_labels: list of ground truth relation labels
        - gt_object_labels: list of ground truth object labels
        - gt_boxes: list of ground truth bounding boxes

    Args:
        prediction (str | Path): _description_
        data (str): _description_
        sg_parser (SceneGraphParser): _description_
        sg_mode (int): _description_
        top_k (int): _description_

    Returns:
        _type_: _description_
    """
    sg_parse = partial(sg_parser.parse_outputs, sg_mode=sg_mode, top_k=top_k) # function to parse scene graph

     # hdf5 output file is from Robin generated scene graph
    if is_hdf5_file(prediction): 
        from osprey.eval.draw_utils import xywh2xyxy
        hdf5_data: dict = load_hdf5_file(prediction)
        df = pd.DataFrame(hdf5_data.values())
        df['regions'] = df['regions'].apply(lambda x: [ast.literal_eval(r) for r in x])
        df['pred_boxes'] = df['regions'].apply(lambda x: [xywh2xyxy(r['bbox']) for r in x])
    else: # jsonl file
        df = pd.read_json(prediction,lines=True)
        from_gpt = 'updated_sg' in df.columns
        if from_gpt: # From GPT generations
            sg_parse = partial(sg_parser.parse_outputs, sg_mode=0, top_k=top_k) # always use mode 0
            df = process_gpt_sg(df)
        else: # From osprey generations
            assert sg_mode == 2, "Only mode 2 is supported for osprey generations"
            # sg_parse = partial(sg_parser.parse_outputs, sg_mode=2, top_k=top_k) # always use mode 2
            df['scene_graph_raw_text'] = df['pred_raw']

    # Parse Scene Graph
    df = df.iloc[:n_samples] if n_samples else df
    df['result'] = df['scene_graph_raw_text'].progress_apply(sg_parse)
    df = df[~df['result'].isnull()]
    for k in ['pred_names', 'pred_labels', 'pred_object_names', 'pred_object_labels', 'pred_triplets']:
        df[k] = df['result'].apply(lambda x: x[k] if x is not None else None)

    # If gt_object_labels not found, assume gt objects is the same as prediction, e.g. for predicate classification
    if 'gt_labels' not in df:
        df = add_gt_data(df, data)
    if 'gt_object_labels' not in df: # Assume object labels are the same as prediction
        df['gt_object_labels'] = df['pred_object_labels']

    print('Number of relationships per image:')
    print(df['pred_labels'].str.len().describe())
    
    return df
    
    
if __name__ == '__main__':

    ''' 

    # hdf5 generations
    python osprey/eval/psg/get_scene_graph_metric.py \
        region_results/psg_test/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_stage_3-mistral-7b-instruct-v0.2_bs16_epoch2/sam_whole_part_sam2_sg_detailed_temp0.5_top0.95_1_0.hdf5 \
        --top_k 200

    # gpt generations
    python osprey/eval/psg/get_scene_graph_metric.py \
        region_results/psg_test/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_stage_3-mistral-7b-instruct-v0.2_bs16_epoch2/sam_whole_part_sam2_sg_detailed_temp0.5_top0.95_1_0_gpt_sg.jsonl \
        --top_k 200

    # robin e2e generations
    python osprey/eval/psg/get_scene_graph_metric.py \
        osprey/eval/results/psg/psg_sg_detection_e2e/osprey_sg_vg_psg_merged_full_v4_mode_2_box-osprey_stage_3-mistral-7b-instruct-v0.2-lr2e-5_bs16_epoch3/temp0.5_top1.0_max_regions_99.jsonl

    ###
    # model-based evaluations
    ### 
    
    # GPT4
    python osprey/eval/psg/get_scene_graph_metric.py \
        region_results/psg_test/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_stage_3-mistral-7b-instruct-v0.2_bs16_epoch2/sam_whole_part_sam2_sg_detailed_temp0.5_top0.95_1_0_gpt_sg.jsonl \
        --top_k 200 --metric model_gpt4 \
        --image_dir  /net/nfs.cirrascale/mosaic/jamesp/images/coco/ \
        --output_file region_results/psg_test/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_stage_3-mistral-7b-instruct-v0.2_bs16_epoch2/sam_whole_part_sam2_sg_detailed_temp0.5_top0.95_1_0_gpt_sg_metric_model_gpt4o.json
    
    # Qwen2VL
     
    '''

    available_metrics = SceneGraphEvaluator.list_evaluators()
    parser = argparse.ArgumentParser(description='osprey sg evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('prediction', help='path to prediction file',)
    parser.add_argument('--bert', help='path to bert model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--category', help='path to Categories', default='/net/nfs.cirrascale/mosaic/jamesp/Osprey/osprey/eval/psg/psg_category.json')
    parser.add_argument('--data', help='path to gt data', default='/net/nfs.cirrascale/mosaic/jamesp/Osprey/osprey/eval/psg/psg_asv2_val_test.json')
    parser.add_argument('--n_samples', help='n_samples to evaluate', type=int, default=None)

    # metric params
    parser.add_argument("--metric", type=str, nargs='+',  choices=available_metrics, default=['sg_recall'])
    parser.add_argument('--sg_mode', type=int, default=2, help='mode of scene graph generation', choices=[0, 1, 2])
    parser.add_argument('--image_dir', type=str, help='path to image directory')
    parser.add_argument('--top_k', type=int, default=50, help='number of top k relations to consider')
    parser.add_argument('--output_file', type=str, default=None, help='output json file to save results with metric and predictions')
    parser.add_argument("--task_name", type=str, default=None)

    # Region config
    from tqdm import tqdm
    args = parser.parse_args()
    tqdm.pandas()

    sg_parser = SceneGraphParser(
        bert_model=args.bert,
        category=args.category,
    )

    df: pd.DataFrame = get_data(
        prediction=args.prediction,
        data=args.data,
        sg_parser=sg_parser,
        sg_mode=args.sg_mode,
        top_k=args.top_k,
        n_samples=args.n_samples 
    )

    # Get bbox coverage
    df['coverage'] = df.progress_apply(lambda x: sg_parser.get_bbox_coverage(x['gt_boxes'], x['pred_boxes']), axis=1)
    coverage = df['coverage'].mean()
    print('Coverage:', coverage)
    
    # Calculate recall / mean recall
    data: list[dict] = df.to_dict(orient='records')
    category = json.load(open(args.category))
    metrics = {'coverage': coverage}
    for metric in args.metric:
        evaluator: SceneGraphEvaluator = SceneGraphEvaluator.get_evaluator(metric)
        result: dict = evaluator(data, category=category, 
                                 image_dir=args.image_dir,
                                 return_predictions=True,
                                 )
        metrics.update(result)

    if args.output_file:
        save_metrics(args.output_file, metrics)