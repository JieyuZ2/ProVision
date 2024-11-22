''' Script to merge regions from different region proposals.'''
from copy import deepcopy
import torch 
import numpy as np
from tqdm import tqdm
import ast
import argparse
import pandas as pd

from osprey.eval.utils import load_hdf5_group, load_hdf5_file, save_results_to_hdf5

if __name__ == '__main__':
    '''
    python -m osprey.eval.annotate.gather_files \
        --prefix region_results/vg_test \
        --output region_results/vg_test/sam_whole_part_sam2.hdf5
    '''
    parser = argparse.ArgumentParser(description='osprey sg generator', formatter_class=argparse.RawTextHelpFormatter)
    # Model config
    parser.add_argument('--region_dir', help='image json file', required=True,) 
    parser.add_argument('--postfix', help='postfix to use', default=None,) 
    parser.add_argument('--n_samples', help='number of images to process', default=None,) 
    parser.add_argument('--output', help='hdf5 or directory to save results', required=True)
    args = parser.parse_args()

    # Load region predictions
    sam_regions = {}
    sam_files = []
    for region_mode in ['sam_whole', 'sam_part', 'sam2']:
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
        regions3 = process_regions(sam_regions['sam2'][image_id]['regions'])

        # Merge regions1 and regions2 with mode 0.
        # add all regions that don't overlap with regions1 or regions that greater than the bottom top 25% of region1 areas
        regions2_merged = get_regions2_to_merge(regions1, regions2, 0)
        regions12_merged = regions1 + regions2_merged
        
        # Merge regions12_merged and regions3 with mode 3.
        # add regions that don't overlap with regions1 and greater than the bottom top 25% of region1 areas
        regions3_merged = get_regions2_to_merge(regions12_merged, regions3, 3)
        regions123_merged = regions12_merged + regions3_merged
        
        item = deepcopy(sam_regions['sam_whole'][image_id])
        item['regions'] = regions123_merged
        merged.append(item) 
    save_results_to_hdf5(merged, args.output, write_mode='w')    
    print(f'Saved {len(merged)} images to {args.output}')
        



    
    