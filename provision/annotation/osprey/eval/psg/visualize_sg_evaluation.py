'''
Evaluation for SG generation
Measures recall and mean recall for subject, object and predicate.
'''
import json
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import supervision as sv

from osprey.eval.draw_utils import visualize_masks
from osprey.eval.vis_utils import convert_df_to_html
# from osprey.verifier import Verifier

tqdm.pandas()
   
def draw_mask(image: Image.Image, detections: np.ndarray) -> Image.Image:
    # Assuming visualize_masks is a function that draws a mask for a single detection
    detections = sv.Detections(detections)
    image_rgb = np.asarray(image)
    image_mask_rgb = visualize_masks(image_rgb, detections, draw_bbox=True, draw_mask=True, white_padding=50, plot_image=False)
    image_mask = Image.fromarray(image_mask_rgb)
    return image_mask

# def draw_masks(image_rgb: np.ndarray, detections_list: list[sv.Detections]) -> list[Image.Image]:
#     ''' Draw masks for detections '''
#     disable_tqdm = True
#     n_processes = 16 # mp.cpu_count()
#     with mp.Pool(processes=n_processes) as pool:
#         results = list(tqdm(
#             pool.starmap(
#                 _draw_mask, 
#                 [(image_rgb, detection) for detection in detections_list]), 
#             total=len(detections_list), disable=disable_tqdm
#         ))
#     return results

if __name__ == '__main__':
    '''
    Example usage:

    python osprey/eval/psg/visualize_sg_evaluation.py \
        --data region_results/psg_test/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_stage_3-mistral-7b-instruct-v0.2_bs16_epoch2/sam_whole_part_sam2_sg_detailed_temp0.5_top0.95_1_0_gpt_sg_metric_model_gpt4o.json \
        --image_dir  /net/nfs.cirrascale/mosaic/jamesp/images/coco/ \
        --output region_results/psg_test/osprey_multi_region_v7_gqa_cot_osprey_stage3_llava_pointqa-osprey_stage_3-mistral-7b-instruct-v0.2_bs16_epoch2/sam_whole_part_sam2_sg_detailed_temp0.5_top0.95_1_0_gpt_sg_metric_model_gpt4o.html
    
    '''
    parser = argparse.ArgumentParser(description='Evaluate scene graph generation')
    parser.add_argument('--evaluator', type=str, default='model_gpt4', help='metric to evaluate')
    parser.add_argument('--data', type=str, help='json file with scene graph evaluation')
    parser.add_argument('--image_dir', type=str, help='Path to image directory')
    parser.add_argument('--output', type=str, help='Path to output file')
    args = parser.parse_args()

    with open(args.data, 'r') as f:
        data = json.load(f)
    
    assert args.evaluator in data, f'{args.evaluator} not in {data.keys()}'
    assert 'predictions' in data[args.evaluator], f'predictions not in {data[args.evaluator].keys()}'
    
    predictions = data[args.evaluator]['predictions']
    df = pd.DataFrame(predictions)
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(args.image_dir, x))
    df['detections'] = df.apply(lambda x: np.array([x['subj_bbox'], x['obj_bbox']]), axis=1)

    # draw masks
    df['image'] = df['image_path'].progress_apply(Image.open)
    df['image_mask'] = df.progress_apply(lambda x: draw_mask(x['image'], x['detections']), axis=1)
    df['triplet'] = df.apply(lambda x: f'{x["subj"]} - {x["pred"]} - {x["obj"]}', axis=1)
    html = convert_df_to_html(
        df[['image_mask', 'triplet', 'answer']], 
        image_keys=['image_mask'],
        max_width=1000
    )
    print(f'Saving visualization to to {args.output}')
    with open(args.output, 'w') as f:
        f.write(html)
