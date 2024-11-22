import argparse
from pathlib import Path
import torch
import os
import json
from tqdm import tqdm
import pandas as pd
from osprey.train.train import DataArguments

from osprey.eval.summary.eval_summary import SummaryEval

import numpy as np
from PIL import Image
import base64
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import argparse

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

def get_mask(mask) -> np.ndarray:
    ''' Returns np bool mask from base64 string'''
    return np.array(Image.open(BytesIO(base64.b64decode(mask))))

class DCISummaryEval(SummaryEval):
    ''' 
    Densely Captioned Image Summarization Evaluation
    '''

    def eval(self, root_path, ann_file, temperature=0.2, top_p=1.0):

        sgs = []
        ann = json.load(open(ann_file))

        data = self.get_chunk(ann)

        for image_id, datum in tqdm(data):
            height = datum['height']
            width = datum['width']

            # Gather regions
            mask_data: dict = datum['mask_data']
            mask_keys = mask_data.keys()
            boxes = []
            segs = []
            for mask_key in mask_keys:
                mask = mask_data[mask_key]

                bounds = mask['bounds']
                bbox = [bounds['topLeft']['x'], bounds['topLeft']['y'], bounds['bottomRight']['x'], bounds['bottomRight']['y']]
                boxes.append(bbox)
                seg = get_mask(mask['outer_mask'])
                segs.append(seg)
            boxes = np.array(boxes)[:self.max_regions]
            segs = np.array(segs)[:self.max_regions]

            # Load GT Summary
            gt_summary: str = datum['short_caption'] + ' ' + datum['extra_caption']
            
            img_path = os.path.join(root_path, datum['image'])

            # Create masks and regions
            masks = self.create_masks(boxes, segs, height, width)
            masks = torch.from_numpy(segs)

            # Generate Grounded Summary
            pred_summary: str = self.generate(img_path=img_path, masks=masks, temperature=temperature, top_p=top_p)

            print("[GT]")
            print(gt_summary)

            print("[Pred]")
            print(pred_summary)

            # Parse prediction
            # pred_summary = self.parse_summary(outputs) 

            sgs.append({
                'image_id': image_id,
                'bboxes': boxes,
                'pred_summary': pred_summary,
                'width': width, 
                'height': height,
                'gt_summary': gt_summary,
            })

        return sgs

if __name__ == "__main__":
    '''
        python -m osprey.eval.summary.eval_dci_summary --model exp/relation_description_summary_coco_sam_seem_box_segm/checkpoint-6000/ \
            --temperature 0.5 \
            --top_p 0.95 \
            --output osprey/eval/results/summary/dci_test/relation_description_summary_coco_sam_seem-gt_objects_temp0.5_top0.95.jsonl
    '''
    parser = argparse.ArgumentParser(description='osprey sg evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--bert', help='path to bert model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--json', help='path to gqa json file with regions', default='data/summary/dci_test.json')
    parser.add_argument('--img', help='path to gqa imgs', default='../DCI/data/images/sa_000138')
    parser.add_argument('--region_mode', type=str, default='segmentation', choices=['box', 'segmentation', 'box_segmentation'])
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print('Saving results to {}'.format(args.output))
    relation_eval = DCISummaryEval(args.model, 
                     debug=args.debug,
                     region_mode=args.region_mode,
                     chunk_idx=args.chunk_idx,
                     num_chunks=args.num_chunks,
                    )
    results = relation_eval.eval(args.img, args.json, args.temperature, args.top_p)
    os.makedirs(Path(args.output).parent, exist_ok=True)
    pd.DataFrame(results).to_json(args.output, orient='records', lines=True)
    print('Saved results to {}'.format(args.output))

