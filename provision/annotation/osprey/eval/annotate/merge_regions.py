''' Script to merge regions from different region proposals.'''
from copy import deepcopy
import torch 
import numpy as np
from tqdm import tqdm
import ast
import argparse

from osprey.eval.utils import load_hdf5_group, load_hdf5_file, save_results_to_hdf5
from osprey.eval.draw_utils import visualize_masks, xywh2xyxy, annToMask, detections_from_sam

def _upcast(t: torch.Tensor) -> torch.Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()
    
def get_box_union(boxes1, boxes2):
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]     
    
    return inter   


def get_regions2_to_merge(regions1, regions2, mode: int, iou_threshold: float = 0.7, non_overlap_threshold: float = 0.6):
    ''' 
    Determine which regions in regions2 should be merged into regions1.
        mode: 
            - 0: add all regions that don't overlap with regions1 or regions that greater than the bottom top 25% of region1 areas
            - 1: only add regions that don't overlap with regions1
            - 2: only add regions that greater than the bottom top 25% of region1 areas
            - 3: add regions that don't overlap with regions1 and greater than the bottom top 25% of region1 areas
    ''' 

    if len(regions1) == 0:
        return regions2
    if len(regions2) == 0:
        return []

    # get the intersection of the masks
    regions1_mask = np.array([annToMask(region['segmentation']) for region in regions1]) # [M x H x W]
    regions2_mask = np.array([annToMask(region['segmentation']) for region in regions2]) # [N x H x W]
    regions1_area = np.sum(regions1_mask, axis=(1, 2))
    regions2_area = np.sum(regions2_mask, axis=(1, 2))

    # Calculate the intersection of each pair of masks
    # We need to expand dimensions to allow broadcasting:
    intersection_mask = np.logical_and(regions1_mask[:, np.newaxis, :, :], regions2_mask[np.newaxis, :, :, :])
    intersection_area = np.sum(intersection_mask, axis=(2, 3))

    # Calculate union areas for each pair
    union_area = regions1_area[:, np.newaxis] + regions2_area[np.newaxis, :] - intersection_area

    # Area threshold
    area_threshold = np.percentile(regions1_area, 50)
    area2_threshold = np.percentile(regions2_area, 75)
    max_regions2_overlap_with_regions = np.max(intersection_area / regions2_area, axis=0)

    # Identify regions in regions2 that should be merged into regions1
    non_overlap_indices = np.where(max_regions2_overlap_with_regions < non_overlap_threshold)[0].tolist()
    area_indices = np.where(regions2_area > area_threshold)[0].tolist()
    area2_indices = np.where(regions2_area > area2_threshold)[0].tolist()

    # ignore high overlaps
    iou = intersection_area / union_area
    max_iou = np.max(iou, axis=0)
    overlapping_indices = np.where(max_iou > iou_threshold)[0]

    if mode == 0: # add all regions that don't overlap with regions1 or regions that greater than the bottom top 25% of region1 areas
        final_indices = sorted(list(set(non_overlap_indices + area_indices + area2_indices)))
    elif mode == 1: # only add regions that don't overlap with regions1 
        final_indices = non_overlap_indices
    elif mode == 2: # only add regions that greater than the bottom top 25% of region1 areas
        final_indices = area_indices
    elif mode == 3:
        final_indices = sorted(list(set(area_indices) & set(non_overlap_indices)))
    elif mode == 4:
        final_indices = sorted(list(set(area2_indices) & set(non_overlap_indices)))
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    final_indices = [i for i in final_indices if i not in overlapping_indices]
    
    regions2_to_add = [regions2[i] for i in final_indices]

    return regions2_to_add

def process_regions(regions):
    return [ast.literal_eval(region) for region in regions]

if __name__ == '__main__':
    '''
    python -m osprey.eval.annotate.merge_regions \
        --region_dir annotate_results/region_results/vg_test \
        --output annotate_results/region_results/vg_test/sam_whole_part_sam2.hdf5
    
    python -m osprey.eval.annotate.merge_regions \
        --region_dir annotate_results/region_results/hico_det_test_1000 \
        --region_mode sam_whole sam_part \
        --output annotate_results/region_results/hico_det_test_1000/sam_whole_part.hdf5
    
    python -m osprey.eval.annotate.merge_regions \
        --region_dir annotate_results/region_results/openimages_test_v6 \
        --region_mode sam_whole sam_part \
        --output annotate_results/region_results/openimages_test_v6/sam_whole_part.hdf5
    '''
    parser = argparse.ArgumentParser(description='osprey sg generator', formatter_class=argparse.RawTextHelpFormatter)
    # Model config
    parser.add_argument('--region_dir', help='image json file', required=True,) 
    parser.add_argument('--region_mode', type=str, nargs='+', default=['sam_whole', 'sam_part', 'sam2'], choices=['sam_whole', 'sam_part', 'sam2']) 
    parser.add_argument('--postfix', help='postfix to use', default=None,) 
    parser.add_argument('--n_samples', help='number of images to process', default=None,) 
    parser.add_argument('--output', help='hdf5 or directory to save results', required=True)
    args = parser.parse_args()

    # Load region predictions
    sam_regions = {}
    sam_files = []
    for region_mode in args.region_mode:
        sam_file = f'{args.region_dir}/{region_mode}.hdf5'
        if args.postfix:
            sam_file = sam_file.replace('.hdf5', f'_{args.postfix}.hdf5')
        print('Loading ', sam_file)
        sam: dict = load_hdf5_file(sam_file)
        sam_regions[region_mode] = sam
        sam_files.append(sam_file)
    assert args.output not in sam_files, f'Output file {args.output} already exists in {sam_files}'
    
    # Get common image ids
    image_ids = [set(sam_regions[mode].keys()) for mode in sam_regions.keys()]
    image_ids = list(set.intersection(*image_ids))
    print('Number of common image ids: ', len(image_ids))

    # Merge regions
    merge_strategy = [0, 3]
    merged = []
    for image_id in tqdm(image_ids[:args.n_samples]):
        regions1 = process_regions(sam_regions['sam_whole'][image_id]['regions'])
        regions2 = process_regions(sam_regions['sam_part'][image_id]['regions'])

        # Merge regions1 and regions2 with mode 0.
        # add all regions that don't overlap with regions1 or regions that greater than the bottom top 25% of region1 areas
        regions2_merged = get_regions2_to_merge(regions1, regions2, 0)
        regions_merged = regions1 + regions2_merged
        
        # Merge regions12_merged and regions3 with mode 3.
        # add regions that don't overlap with regions1 and greater than the bottom top 25% of region1 areas
        if 'sam2' in args.region_mode:
            regions3 = process_regions(sam_regions['sam2'][image_id]['regions'])
            regions3_merged = get_regions2_to_merge(regions_merged, regions3, 3)
            regions_merged = regions_merged + regions3_merged
        
        item = deepcopy(sam_regions['sam_whole'][image_id])
        item['regions'] = regions_merged
        merged.append(item) 
    save_results_to_hdf5(merged, args.output, write_mode='w')    
    print(f'Saved {len(merged)} images to {args.output}')
        



    
    