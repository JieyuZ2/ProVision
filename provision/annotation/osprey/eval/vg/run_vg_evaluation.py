import torch
import pandas as pd

import argparse
from tqdm import tqdm
from typing import List, Tuple, Dict
import re

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation.vg.vg_eval import evaluate_relation_of_one_image
from maskrcnn_benchmark.data.datasets.visual_genome import VGDataset, load_info
from maskrcnn_benchmark.utils.logger import setup_logger

from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.data.datasets.evaluation.vg.sgg_eval import SGRecall, SGNoGraphConstraintRecall, SGZeroShotRecall, SGNGZeroShotRecall, \
    SGPairAccuracy, SGMeanRecall, SGNGMeanRecall


def get_evaluator(mode: str,  num_rel_category: int, ind_to_predicates):
    result_dict = {}
    evaluator = {}
    # tradictional Recall@K
    eval_recall = SGRecall(result_dict)
    eval_recall.register_container(mode)
    evaluator['eval_recall'] = eval_recall

    # no graphical constraint
    eval_nog_recall = SGNoGraphConstraintRecall(result_dict)
    eval_nog_recall.register_container(mode)
    evaluator['eval_nog_recall'] = eval_nog_recall

    # test on different distribution
    eval_zeroshot_recall = SGZeroShotRecall(result_dict)
    eval_zeroshot_recall.register_container(mode)
    evaluator['eval_zeroshot_recall'] = eval_zeroshot_recall

    # test on no graph constraint zero-shot recall
    eval_ng_zeroshot_recall = SGNGZeroShotRecall(result_dict)
    eval_ng_zeroshot_recall.register_container(mode)
    evaluator['eval_ng_zeroshot_recall'] = eval_ng_zeroshot_recall

    # used by https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
    eval_pair_accuracy = SGPairAccuracy(result_dict)
    eval_pair_accuracy.register_container(mode)
    evaluator['eval_pair_accuracy'] = eval_pair_accuracy

    # used for meanRecall@K
    eval_mean_recall = SGMeanRecall(result_dict, num_rel_category, ind_to_predicates, print_detail=True)
    eval_mean_recall.register_container(mode)
    evaluator['eval_mean_recall'] = eval_mean_recall

    # used for no graph constraint mean Recall@K
    eval_ng_mean_recall = SGNGMeanRecall(result_dict, num_rel_category, ind_to_predicates, print_detail=True)
    eval_ng_mean_recall.register_container(mode)
    evaluator['eval_ng_mean_recall'] = eval_ng_mean_recall
    
    return evaluator

def finalize_evaluation(evaluator, mode: str):
    # This is a placeholder for your logic to summarize and output the evaluation results
    # You might want to loop through your evaluators and call any final calculation or print methods
    for key, eval_obj in evaluator.items():
#         if hasattr(eval_obj, 'summarize'):
        print(eval_obj.generate_print_string(mode))

def convert_dict_to_boxlist(dct: dict, xyxy, width, height) -> BoxList:
    box = BoxList(bbox=xyxy, image_size=(width,height))
    for k,v in dct.items():
        box.add_field(k,v)
    return box

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_results', required=True)
    parser.add_argument('--mode', default='sgcls', choices=['sgcls', 'sgdet'])

    args = parser.parse_args()

    mode = args.mode

    logger = setup_logger('default', save_dir=None, distributed_rank=0)
    zeroshot_triplet = torch.load("maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_triplet.pytorch", map_location=torch.device("cpu")).long().numpy()

    print('Loading dataset...')
    ind_to_classes, ind_to_predicates, ind_to_attributes = load_info("datasets/vg/VG-SGG-dicts-with-attri.json")
    if args.mode == 'sgcls':
        cfg.merge_from_file('config_sgcls.yml')
    elif args.mode == 'sgdet':
        cfg.merge_from_file('config_sgdet.yml')
    else:
        raise ValueError("Unknown sg mode: {}".format(args.mode))
    print('Loaded Dataset')
    
    # Assuming cfg and logger are globally accessible or passed as arguments
    # You need to adjust the cfg attributes access according to your configuration structure
    attribute_on = cfg.MODEL.ATTRIBUTE_ON
    num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
    num_rel_category = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
    multiple_preds = cfg.TEST.RELATION.MULTIPLE_PREDS
    iou_thres = cfg.TEST.RELATION.IOU_THRESHOLD

    # Prepare the global container with necessary information
    global_container = {
        'attribute_on': attribute_on,
        'num_attributes': num_attributes,
        'num_rel_category': num_rel_category,
        'multiple_preds': multiple_preds,
        'iou_thres': iou_thres,
        'mode': mode,
        'zeroshot_triplet': zeroshot_triplet
    }

    # Initialize evaluators
    evaluator = get_evaluator(mode, num_rel_category, ind_to_predicates, )

    # Load predictions
    preds: dict = {d['image_id']: d for d in torch.load(args.eval_results)}

    # Load GT Annotations
    gts: Dict = torch.load('vg_test_img_gt.pytorch')

    # Loop through groundtruths and predictions to evaluate
    for img_id, pred_item in list(preds.items()):

        # gt_labels: dict = pred_item['gt_labels']
        pred_labels: dict = pred_item['prediction_labels']
        
        # convert to boxlist ...
        gt = gts[img_id]
        pred = convert_dict_to_boxlist(pred_labels, pred_item['bboxes'], pred_item['width'], pred_item['height'])
        gt = convert_dict_to_boxlist(gt, gt['bbox'], gt['width'], gt['height'])
        evaluate_relation_of_one_image(gt, pred, global_container, evaluator)
        
    # calculate mean recall
    evaluator['eval_mean_recall'].calculate_mean_recall(mode)
    evaluator['eval_ng_mean_recall'].calculate_mean_recall(mode)

    # Assuming you have a function to finalize and print/return evaluation results
    # This part needs to be implemented based on how you want to summarize and output your evaluation results
    finalize_evaluation(evaluator, mode)

