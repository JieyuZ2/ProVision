import json
import os

if __name__ == "__main__":
    
    image_dir = '../images/coco/' 
    input_file = 'osprey/eval/stage3_eval/paco_val_1k_category.json'
    j = json.load(open(input_file))
    for d in j:
        if os.path.isfile(os.path.join(image_dir, 'train2017', d['file_name'])):
            d['file_name'] = os.path.join('train2017', d['file_name'])
        elif os.path.isfile(os.path.join(image_dir, 'val2017', d['file_name'])):
            d['file_name'] = os.path.join('val2017', d['file_name'])
        else:
            print('Failed to find file: {}'.format(d['file_name']))
    breakpoint()

    output_file = input_file
    json.dump(j, open(output_file, 'w'))

