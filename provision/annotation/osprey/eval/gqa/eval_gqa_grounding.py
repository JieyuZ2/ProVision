''' Assumes GT regions is used'''
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torchvision.ops.boxes import box_iou
import re
import ast

def find_best_assignments(ious: torch.Tensor, placeholder=0) -> torch.LongTensor:
    """
        Finds the best assignment of candidate bounding boxes to multiple reference bounding boxes based on the
        Intersection over Union (IoU) values. The function iteratively selects the candidate with the highest IoU
        for each reference bounding box, ensuring that each candidate is uniquely assigned. If the number of candidate
        bounding boxes is less than the number of reference bounding boxes, a placeholder value is used for the
        unmatched references.

        Parameters:
        - ious (torch.Tensor): A 2D tensor of shape (B, N) containing the IoU values between each candidate bounding box
        and each reference bounding box, where B is the number of candidate bounding boxes and N is the number of
        reference bounding boxes.
        - placeholder (int, optional): The placeholder value to use for unmatched reference bounding boxes if there are
        fewer candidates than references. Defaults to -1.

        Returns:
        - torch.Tensor: A 1D tensor of length N containing the indices of the candidate bounding boxes that have
        the highest IoU with each reference bounding box. If a reference bounding box cannot be matched due to
        insufficient candidates, its index is set to the placeholder value.
    """

    ious = torch.clone(ious)
     
    num_candidates, num_refs = ious.shape
    selected_candidates = torch.ones(num_refs, dtype=torch.long) * placeholder
    available_candidates = torch.arange(num_candidates)
    
    for _ in range(min(num_refs, num_candidates)):
        # Find the global maximum IoU value and its indices in the remaining matrix
        max_iou_value, flat_max_index = ious[available_candidates].flatten().max(0)
        max_candidate_index, max_ref_index = divmod(flat_max_index.item(), num_refs)
        
        # Append the selected candidate index to the result list
        selected_candidates[max_ref_index] = max_candidate_index

        # assign low iou value if taken
        ious[max_candidate_index] = -1 # ious[max_candidate_index]
    
    return selected_candidates

def eval_grounding(results: list):
    eval_result = ([_eval_grounding(r) for r in results])
    eval_result = ([r for r in eval_result if r is not None])

    # Calculate averages
    eval_metric = {}
    for k in eval_result[0].keys():
        v = eval_result[0][k]
        if isinstance(v, np.ndarray) or isinstance(v, list):
            v = sum(np.mean(r[k]) for r in eval_result) / len(eval_result)
        else:
            v = sum(r[k] for r in eval_result) / len(eval_result)
        eval_metric[k] = v
    
    return eval_metric, eval_result

def _eval_grounding(result: dict):

    annot = annots[result['question_id']]
    candidate_regions = annot['objects']
    object_mapping = annot['object_mapping']
    object_ids = annot['data']['object_ids']
    pred_sg = parse_sg(result['output']) # {'region_id': info}

    gt_regions = [candidate_regions[object_mapping[d]]['bbox'] for d in object_ids]
    ref_bboxes = torch.Tensor(gt_regions)
    if pred_regions is not None:
        candidate_regions = pred_regions[result['question_id']]['regions']
        if True:
            candidate_regions = sorted([d for d in candidate_regions], key=lambda x: -x['area'])
        bboxes = torch.Tensor([candidate_regions[int(r)-1]['xyxy'] for r in pred_sg if int(r)-1 < len(candidate_regions)])
    else:
        bboxes = torch.Tensor([candidate_regions[str(int(r)-1)]['bbox'] for r in pred_sg if int(r)-1 < len(candidate_regions)])
    
    if len(bboxes) == 0:
        best_ious = np.zeros(len(gt_regions))
    if len(ref_bboxes) == 0:
        return None
    else:
        # Avg number of Recall
        try:
            ious = box_iou(bboxes, ref_bboxes) # [B x N]
            indices = find_best_assignments(ious) # [N]
            best_ious = ious[indices, torch.arange(len(indices))].numpy()
        except IndexError:
            best_ious = np.zeros(len(gt_regions))
            print('Failed for: {}'.format(pred_sg))
            # breakpoint()

    return {
        'correct': result['pred_answer'] == result['answer'],
        'ious': best_ious,
        'iou_thresh_0.5': best_ious > 0.5,
    }

def parse_sg(input_str):
    # Regular expression to match the region pattern
    pattern = r"region(\d+): ({.*?})\n?"
    matches = re.findall(pattern, input_str, re.DOTALL)
    
    regions = {}
    for match in matches:
        region_id, dict_str = match
        # Safely evaluate the dictionary string
        try:
            dict_data = ast.literal_eval(dict_str)
            regions[region_id] = dict_data # 1-indexed
        except Exception as e:
            print('Failed to parse: {}'.format(dict_str))
            continue
    
    return regions



if __name__ == "__main__":
    '''
        python osprey/eval/gqa/eval_gqa_grounding.py \
            --input osprey/eval/results/gqa/val_balanced_aokvqa/gqa_regions/gqa_cot_sg_relevant_left_right_v3_stage3_sam_hq-latest-temp0.2_top0.95.jsonl \
            --jsonl ./data/gqa/val_balanced_aokvqa_cot_gqa_sam_hq.jsonl
        
        python osprey/eval/gqa/eval_gqa_grounding.py \
            --input osprey/eval/results/gqa/val_balanced_aokvqa/sam_regions/gqa_cot_sg_relevant_left_right_v3_stage3-latest-temp0.2_top1.0.jsonl \
            --pred_regions ./data/gqa/val_balanced_aokvqa_sam_pred_regions.jsonl \
            --jsonl ./data/gqa/val_balanced_aokvqa_cot_gqa_sam_hq.jsonl
        
            
    '''
    parser = argparse.ArgumentParser(description='osprey CoT gqa evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input', help='results file generated by eval code', required=True)
    parser.add_argument('--pred_regions', help='path to jsonl file with predicted regions')
    parser.add_argument('--jsonl', help='path to jsonl file with GT regions', default='./data/gqa/val_cot_gqa_sam_hq.jsonl')
    parser.add_argument('--sort_largest', action='store_true')
    args = parser.parse_args()

    annots = pd.read_json(args.jsonl, lines=True)
    annots['question_id'] = annots['data'].apply(lambda x: int(x['question_id']))
    annots: dict = annots.set_index('question_id').T.to_dict()

    data = pd.read_json(args.input, lines=True)
    data = data[data['question_id'].isin(annots)]
    pred_regions = None
    if args.pred_regions is not None:
        pred_regions = pd.read_json(args.pred_regions, lines=True).set_index('question_id').T.to_dict()
        data = data[data['question_id'].isin(pred_regions)]
    
    data: list = data.to_dict(orient='records')
    print(len(data))
    
    eval_metric, eval_result = eval_grounding(data)
    print(eval_metric)
    eval_result = pd.DataFrame(eval_result)

    correct_grounding = eval_result[eval_result['iou_thresh_0.5'].apply(all)]
    print('Accuracy with correct grounding ({} / {}): {}'.format(len(correct_grounding), len(eval_result) ,correct_grounding['correct'].mean()))



