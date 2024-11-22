import argparse
import os
import pandas as pd
from PIL import Image

from eval import WandbVisualizer

def display_row(row):

    image_path = os.path.join(args.img, str(row['image_id'])+'.jpg')
    summary = row['pred_summary']
    boxes = row['bboxes'] # xyxy bboxes

    im = Image.open(image_path)

    bbox_image = WandbVisualizer.draw_box_to_image(im, boxes, summary)

    return {
        'image_id': row['image_id'],
        'width': row['width'],
        'height': row['height'],
        'summary': summary, 
        'image': bbox_image
    }


if __name__ == "__main__":

    '''
        python osprey/eval/relation/visualize_summary.py \
            --input_file osprey/eval/results/summary/psg_test/relation_description_summary_coco_sam_seem-gt_objects_temp0.5_top0.95.jsonl
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help='path to summary predictions')
    parser.add_argument('--img', help='path to gqa imgs', default='/mmfs1/gscratch/raivn/jspark96/data/images/gqa/images')

    args = parser.parse_args()

    df = pd.read_json(args.input_file, lines=True)

    display_df = pd.DataFrame(df.apply(display_row, axis=1))

    breakpoint()

