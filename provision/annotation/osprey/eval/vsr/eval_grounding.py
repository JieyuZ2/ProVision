import argparse
import os
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

    # Calculate averages
    eval_metric = {}
    for k in eval_result[0].keys():
        v = sum(r[k] for r in eval_result) / len(eval_result)
        if isinstance(v, torch.Tensor):
            v = v.item()
        eval_metric[k] = v
    
    return eval_metric, eval_result

def _eval_grounding(result: dict):

    annot = annots[result['question_id']]
    subj = annot['subj']
    obj = annot['obj']

    candidate_regions = annot['regions']
    pred_regions = parse_regions(result['output']) # {'region_id': info}

    has_subj = False
    has_obj = False

    for region in pred_regions.values():
        if region['name'] == subj:
            has_subj = True
        if region['name'] == obj:
            has_obj = True
    
    # if not has_subj:
    #     print('subj wrong', pred_regions, subj)
    # if not has_obj:
    #     print('obj wrong', pred_regions, obj)
    
    subj_bbox = annot['subj_bbox']
    obj_bbox = annot['obj_bbox']

    if 'bbox' in annot:
        candidate_regions = annot['bbox']
        bboxes = torch.Tensor([candidate_regions[int(r)-1] for r in pred_regions if int(r)-1 < len(candidate_regions)])
    else:
        bboxes = torch.Tensor([candidate_regions[int(r)-1]['xyxy'] for r in pred_regions if int(r)-1 < len(candidate_regions)])

    if False: # use all candidate regions
        bboxes = torch.Tensor([r['xyxy'] for r in candidate_regions])
    if len(bboxes) == 0:
        iou_subj = 0
        iou_obj = 0
    else:
        ref_bboxes = torch.Tensor([subj_bbox['xyxy'], obj_bbox['xyxy']])

        try:
            ious = box_iou(bboxes, ref_bboxes) # [B x 2]
            indices = find_best_assignments(ious) # [2]

            # print(ious)
            # print(indices)
            iou_subj = ious[indices[0], 0].item()
            iou_obj = ious[indices[1], 1].item()
        except IndexError:
            breakpoint()
    

    return {
        'correct': result['pred_answer'] == result['answer'],
        'has_subj': has_subj, 
        'has_obj': has_obj,
        'iou_subj': iou_subj,
        'iou_obj': iou_obj,
        'iou_subj_thresh_0.5': iou_subj > 0.5,
        'iou_obj_thresh_0.5': iou_obj > 0.5,
        # 'iou_subj_thresh_0.3': iou_subj > 0.3,
        # 'iou_obj_thresh_0.3': iou_obj > 0.3,
    }

def parse_regions(input_str):
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
        python osprey/eval/vsr/eval_grounding.py \
            --jsonl data/vsr/zeroshot_test_sam_regions.jsonl \
            --input  osprey/eval/results/vsr/stage3_gqa_cot_sg_relevant_left_right_v3-latest-temp0.2_top1.0.jsonl
        
        python osprey/eval/vsr/eval_grounding.py \
            --jsonl data/vsr/zeroshot_test_coco_objects_sam_hq.jsonl \
            --input osprey/eval/results/vsr/coco_regions/gqa_cot_sg_relevant_left_right_v3_stage3-latest-temp0.2_top1.0.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey CoT gqa evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input', help='results file generated by eval code', required=True)
    parser.add_argument('--jsonl', help='path to vsr jsonl file with regions', required=True)
    args = parser.parse_args()

    data: list = pd.read_json(args.input, lines=True).to_dict(orient='records')
    annots: dict = pd.read_json(args.jsonl, lines=True).set_index('index').T.to_dict()

    eval_metric, eval_result = eval_grounding(data)
    print(eval_metric)
    eval_result = pd.DataFrame(eval_result)

    correct_grounding = eval_result[eval_result['iou_subj_thresh_0.5'] & eval_result['iou_obj_thresh_0.5']]
    print('Accuracy with correct grounding: {}'.format(correct_grounding['correct'].mean()))



