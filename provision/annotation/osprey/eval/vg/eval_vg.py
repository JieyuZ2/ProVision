import argparse
from pathlib import Path
import torch
import os
import json
import argparse
from tqdm import tqdm
import pandas as pd
import re
from typing import List, Literal, Dict, Tuple
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.mm_utils import tokenizer_image_token
from osprey.train.train import DataArguments

from osprey.datasets.sg import SGDataset, Ref_WAY, OBJECT_QUESTIONS, SG_QUESTIONS
from osprey.datasets.relation import RelationDataset, RELATION_QUESTIONS
from osprey.eval.eval import OspreyEval

import numpy as np
from sentence_transformers import SentenceTransformer, util

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

def get_region_id(region_str: str) -> int:
    """
    Extracts and converts the region ID from a string to 0-indexed integer.
    """
    match = re.search(r'region(\d+)', region_str)
    if match:
        return int(match.group(1)) - 1
    return -1


class VGEval(OspreyEval, SGDataset):
    ''' 
    VG Scene Graph evaluation class that assigns generation result to scene graph object and relationship class
    using BertModel.
    '''
    def __init__(
            self, 
            model_path, 
            bert_model, 
            category,
            max_regions=99,
            max_attributes_per_obj=5,
            max_relations_per_obj=5,
            region_mode: Literal['box', 'segmentation', 'box_segmentation']='segmentation',
            ignored_relations: List[str] = None,
            generate_relations: bool = False,
            sg_mode: int=3, 
            debug=False, ):
        
        if ignored_relations is None:
            ignored_relations = []
        super().__init__(model_path, debug=debug)
        self.max_regions = max_regions
        self.sg_mode = sg_mode

        self.generate_relations = generate_relations
        self.max_attributes_per_obj = max_attributes_per_obj
        self.max_relations_per_obj = max_relations_per_obj
        self.ignored_relations = ignored_relations
        self.region_mode = region_mode
        self.use_box = region_mode == 'box'
        self.is_train = False

        # Set object and relationship labels
        category_dicts = json.load(open(category))

        # Add background 
        category_dicts['idx_to_label'][0] = 'background'
        category_dicts['idx_to_predicate'][0] = 'none'

        self.idx_to_label: Dict[int, str] = {int(k): v for k,v in category_dicts['idx_to_label'].items()}
        self.label_to_idx: Dict[str, int] = {v: k for k,v in category_dicts['idx_to_label'].items()}
        self.idx_to_predicate: Dict[int, str] = {int(k): v for k,v in category_dicts['idx_to_predicate'].items()}
        self.predicate_to_idx: Dict[str, int] = {v: k for k,v in category_dicts['idx_to_predicate'].items()}

        self.idx_to_attribute: Dict[int, str] = {int(k): v for k,v in category_dicts['idx_to_attribute'].items()}

        # Set Object Embeddings
        self.bert_model = SentenceTransformer(bert_model) 
        object_categories = [self.idx_to_label[k] for k in sorted(self.idx_to_label)]
        self.object_embeddings = self.bert_model.encode(object_categories, convert_to_tensor=True)

    def eval(self, root_path, ann_file, temperature=0.2, top_p=1.0):

        sgs = []
        anns = pd.read_json(ann_file, lines=True).to_dict(orient='records')

        for ann in tqdm(anns[:100]):
            image_id = str(ann['image_id'])
            height = ann['height']
            width = ann['width']
            regions: list = ann['regions'][:self.max_regions]

            # Gather regions
            region_sg = {}
            segs = []
            bboxes = []
            for idx,region in enumerate(regions):
                ''' index 0 <-> region1'''

                name = self.idx_to_label[region['labels']]
                attributes = [self.idx_to_attribute[att] for att in region['attributes'] if att != 0]
                relations = [f"{self.idx_to_predicate[rel]} region{r+1}" for r, rel in enumerate(region['relation']) if rel != 0]
                
                region_id = idx + 1
                sg_dict = {'name': name, 'attributes': attributes, 'relations': relations}
                region_sg[region_id] = sg_dict

                # add region segmentations
                bboxes.append(region['bbox'])
                segs.append(region['segmentation'])
            
            img_path = os.path.join(root_path, image_id+'.jpg')

            # Add region prompts
            num_objects = len(segs)
            ref_string = ''
            for i in range(num_objects):
                ref_string = ref_string +  f'region{i+1} <mask><pos>' + ','
            ref_string = ref_string[:-1]
            ref_prefix = Ref_WAY[0]

            begin_string = ref_prefix.replace('<region>', ref_string)

            # GT scene graph for reference
            gt = self.create_conversation(region_sg, begin_string)
            if self.sg_mode == 2:
                gt_object, gt_relation = gt[1]['value'].split('Relations:')
                gt_relation = 'Relations:' + gt_relation
            elif self.sg_mode == 3:
                gt_object = gt[1]['value']
                gt_relation = gt[3]['value']

            # Process scene graph to create mask and regions
            if self.use_box:
                masks = [self.bboxToMask(bbox,height,width) for bbox in bboxes]
            else:
                masks = [self.annToMask(ann,height,width) for ann in segs]
            masks = np.array(masks)
            masks = torch.from_numpy(masks)

            # Create input prompt
            if self.generate_relations:
                prompt = begin_string + ' ' + RELATION_QUESTIONS[0]
            elif self.sg_mode in [1,2]:
                prompt = begin_string + ' ' + SG_QUESTIONS[0]
            elif self.sg_mode in [3]:
                prompt = begin_string + ' ' + OBJECT_QUESTIONS[0]
            init_inputs = self.get_init_inputs(img_path,
                                        self.image_processor,
                                        masks=masks,
                                        prompt=prompt,
                                        )

            image = init_inputs['image']
            masks = init_inputs['masks'].cuda()

            conv = self.get_new_conv()
            qs = init_inputs['sources'][0][0]['value']

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            outputs: str = self.get_outputs(image, input_ids, masks, self.stop_str, temperature, top_p, max_new_tokens=1024)

            # relationship scoring
            # prompts = []
            # bs = len(self.idx_to_predicate)
            # image_batch = image.unsqueeze(0).repeat(bs, 1, 1, 1)
            # mask_batch = [masks.half() for _ in range(bs)]
# 
            # # First Identify Objects with Notable Relationships.
            # rel_candidates = {}
            # for i in tqdm(range(len(bboxes))):
            #     for j in tqdm(range(len(self.idx_to_predicate))):
            #         pred = self.idx_to_predicate[j]
            #         conv = self.get_new_conv()
            #         conv.append_message(conv.roles[0], qs)
            #         conv.append_message(conv.roles[1], f"Relations:\nregion{i+1}: {pred}")
            #         prompt = conv.get_prompt()
            #         prompts.append(prompt)
                
            #     relation_input_ids = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for prompt in prompts]
            #     relation_input_ids = torch.nn.utils.rnn.pad_sequence(
            #         relation_input_ids,
            #         batch_first=True,
            #         padding_value=self.tokenizer.pad_token_id
            #     ).cuda()
            #     relation_input_ids[relation_input_ids == self.tokenizer.eos_token_id] = self.tokenizer.pad_token_id
            #     labels = relation_input_ids[:,input_ids.size(1):]
            #     scores: torch.Tensor = self.get_scores(image_batch, relation_input_ids, labels, mask_batch)
            #     base_score = scores[0]
            #     rel_candidates[i] = torch.nonzero(scores > base_score).squeeze()
            # breakpoint()

            # # Next, assign the best object.
            # for region_id, idx in rel_candidates.items(): 
            #     for j in range(len(bboxes)):
            #         pred = self.idx_to_predicate[j]
            #         conv = self.get_new_conv()
            #         conv.append_message(conv.roles[0], qs)
            #         conv.append_message(conv.roles[1], f"Relations:\nregion{region_id+1}: {pred} region{j+1}")
            #         prompt = conv.get_prompt()
            #         prompts.append(prompt)
                
            #     relation_input_ids = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for prompt in prompts]
            #     relation_input_ids = torch.nn.utils.rnn.pad_sequence(
            #         relation_input_ids,
            #         batch_first=True,
            #         padding_value=self.tokenizer.pad_token_id
            #     ).cuda()
            #     relation_input_ids[relation_input_ids == self.tokenizer.eos_token_id] = self.tokenizer.pad_token_id
            #     labels = relation_input_ids[:,input_ids.size(1):]
            #     scores: torch.Tensor = self.get_scores(image_batch, relation_input_ids, labels, mask_batch)
            #     bbox_scores.append(scores)

            print("[GT]")
            if self.generate_relations:
                print(gt_relation)
            else:
                print(gt_object)
                print(gt_relation)
            print("[Pred]")
            print(outputs)

            # Parse output
            if self.sg_mode == 2:
                object_outputs, rel_outputs = outputs.split('Relations:')
                prediction_labels: dict = self.get_prediction_labels(object_outputs, rel_outputs)

            # Generate relations in the second stage using predicted/GT objects
            if self.sg_mode == 3:

                conv = self.get_new_conv()
                qs = init_inputs['sources'][0][0]['value']

                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], gt_object if args.use_gt_objects else outputs)
                conv.append_message(conv.roles[0], RELATION_QUESTIONS[0])
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                rel_outputs: str = self.get_outputs(image, input_ids, masks, self.stop_str, temperature, top_p, max_new_tokens=1024)

                # Extract Scene Graph Labels from raw text.
                object_outputs = outputs
                print(rel_outputs)
                prediction_labels: dict = self.get_prediction_labels(object_outputs, rel_outputs)

            gt_labels = self.get_prediction_labels(gt_object, gt_relation)

            sgs.append({
                'image_id': image_id,
                'region_sg': region_sg,
                'bboxes': bboxes,
                'output': outputs,
                'prediction_labels': prediction_labels,
                'width': width, 
                'height': height
            })

        return sgs

    def get_prediction_labels(self, object_outputs: str, relation_outputs: str):
        ''' Extracts object and relation prediction labels from raw string'''

        predictions = {}
        object_preds: List[Dict] = self.get_object_preds(object_outputs)
        object_labels: dict = self.get_object_labels(object_preds)
        predictions.update(object_labels)

        relation_triplets = self.get_relation_triplets(relation_outputs)
        relation_preds: dict = self.get_relation_labels(relation_triplets, object_preds)
        predictions.update(relation_preds)

        return predictions


    def get_object_preds(self, object_outputs: str) -> List[Dict]:
        ''' Extract object predictions from raw string'''

        predictions = []

        def extract_object_and_attributes(line: str) -> dict:
            """
            Parses a line to extract the object and its attributes if present.
            """
            # Splitting the line at "is" if it exists, to separate object from attributes
            parts = line.split(' is ')
            obj = parts[0].strip()
            attributes = parts[1].strip().split(', ') if len(parts) > 1 else []
            return {'object': obj, 'attributes': attributes}

        for output in object_outputs.split('\n'):
            # Extract the region ID
            region_id = get_region_id(output)
            if region_id >= 0:
                # Extract everything after "regionXX: " for processing
                content = re.sub(r'^region\d+:\s*', '', output)
                # Parse the remaining content for object and attributes
                obj_data = extract_object_and_attributes(content)

                index = region_id-1 # 1 -index to 0-index
                predictions.append(obj_data)
    
        return predictions
    
    def get_object_labels(self, object_preds: List[Dict]) -> Dict[str, torch.Tensor]:
        ''' Assign object labels using BertModel '''
        object_pred_str = [f"{', '.join(pred['attributes'])} {pred['object']}" for pred in object_preds]
        cur_obj_emb = self.bert_model.encode(object_pred_str, convert_to_tensor=True)
        pred_logits: torch.Tensor = util.cos_sim(cur_obj_emb, self.object_embeddings)
        pred_scores, pred_obj_class = pred_logits.max(dim=1)
        
        return {
                'pred_labels': pred_obj_class.detach().cpu(),
                'pred_scores': pred_scores.detach().cpu(), 
                'pred_logits': pred_logits.detach().cpu()
            }


    def get_relation_triplets(self, outputs: str) -> List[Tuple]:
        """
        Extract relation string into a list of triplets.
        """

        def get_triplet(relation_str: str) -> Tuple[int, str, int]:
            """
            Extracts the triplet (source, relation, target) from a relation string.
            """
            parts = relation_str.split(':')
            source_id = get_region_id(parts[0])
            if ',' in parts[1]:
                # Handling multiple relations for the same source region
                relations = parts[1].split(',')
                for relation in relations:
                    relation = relation.strip()
                    relation_type, target_str = relation.split('region')
                    target_str = target_str.strip()
                    if target_str.isdigit():
                        target_id = int(target_str)-1
    #                 target_id = get_region_id(target_str)
                        yield (source_id, relation_type, target_id)
            else:
                relation_type, target_str = parts[1].strip().split(' ', 1)
                target_id = get_region_id(target_str)
            yield (source_id, relation_type, target_id)
            
        triplets = []

        relations = outputs.split('\n')
        relations = [rel for rel in relations if 'region' in rel and ':' in rel]

        for rel in relations:
            if rel.count(':') == 1:
                for triplet in get_triplet(rel):
                    triplets.append(triplet)
        return triplets

    def get_relation_labels(self, triplets: List[Tuple], object_preds: List[Dict])-> Dict[str, torch.Tensor]:
        """
        Map relation triplet to relation labels using BertModel.
        """
            
        rel_pair_idxs = []
        pred_rel_scores = []
        pred_rel_labels = []
        relations = torch.zeros((len(object_preds), len(object_preds)), dtype=torch.long)
        
        for triplet in triplets:
            subj, rel, obj = triplet
            
            if subj >= len(object_preds) or obj >= len(object_preds):
                continue
            
            # Extract subject and object label from object_preds
            subj_text = object_preds[subj]['object']
            obj_text = object_preds[obj]['object'] 
            
            triplet_text = f"{subj_text} {rel} {obj_text}"
            candidates = [f"{subj_text} {r} {obj_text}" for idx, r in sorted(self.idx_to_predicate.items())]
            
            # Get closest relationship using BertModel
            cur_rel_emb = self.bert_model.encode([triplet_text], convert_to_tensor=True)
            rel_embeddings = self.bert_model.encode(candidates, convert_to_tensor=True)
            pred_logits = util.cos_sim(cur_rel_emb, rel_embeddings)
            _, pred_rel_label = pred_logits.max(dim=1)
            
            # store data
            if len(pred_logits) > 0:
                relations[subj, obj] = pred_rel_label[0]
                rel_pair_idxs.append([triplet[0], triplet[2]])
                pred_rel_scores.append(pred_logits)
                pred_rel_labels.append(pred_rel_label)
        
        # Add Dummy
        if len(pred_rel_scores) == 0:
            pred_rel_scores = [torch.zeros(1, len(self.idx_to_predicate))]
            pred_rel_labels = [torch.zeros(1, dtype=torch.long)]
        
        return {
            'rel_pair_idxs': torch.LongTensor(rel_pair_idxs),
            'pred_rel_scores': torch.cat(pred_rel_scores).detach().cpu(),
            'pred_rel_labels': torch.cat(pred_rel_labels).detach().cpu(),
            'relations': relations.detach().cpu()
        }


if __name__ == "__main__":
    '''
        python -m osprey.eval.vg.eval_vg --model exp/vg_sg_v3_stage3/checkpoint-8000/ \
            --temperature 0.5 \
            --top_p 0.9 \
            --output osprey/eval/results/sg/test/vg_sg_v3_stage3-8000-gt_objects_temp0.5_top0.9.jsonl

        python -m osprey.eval.vg.eval_vg --model exp/vg_sg_v4_box/ \
            --temperature 0.2 \
            --top_p 0.9 \
            --sg_mode 3 \
            --region_mode box \
            --output osprey/eval/results/sg/test/vg_sg_v4_box-latest-temp0.2_top0.9.pt \

        python -m osprey.eval.vg.eval_vg --model exp/vg_sg_v4_shuffle \
            --temperature 0.5 \
            --top_p 0.9 \
            --output osprey/eval/results/sg/test/vg_sg_v4_shuffle-latest-temp0.5_top0.9.pt
    '''
    parser = argparse.ArgumentParser(description='osprey sg evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', required=True)
    parser.add_argument('--bert', help='path to bert model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--json', help='path to gqa json file with regions', default='data/sg/vg_test_sg_sam_hq_subset.jsonl')# default='data/sg/test_vg_sg_sam_hq.json')
    parser.add_argument('--category', help='path to VG Categories', default='osprey/eval/sg/datasets/vg/VG-SGG-dicts-with-attri.json')
    parser.add_argument('--img', help='path to gqa imgs', default='/mmfs1/gscratch/raivn/jspark96/data/images/gqa/images')
    parser.add_argument('--sg_mode', type=int, default=3)
    parser.add_argument('--region_mode', type=str, default='segmentation', choices=['box', 'segmentation'])
    parser.add_argument('--temperature', help='temperature generation params', type=float, default=0.2)
    parser.add_argument('--top_p', help='top_p generation params', type=float, default=1.0)
    parser.add_argument('--output', help='path to save results to json file', required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_gt_objects', action='store_true', help='used gt objects when generating relations for sg_mode: 3')
    parser.add_argument('--generate_relations', action='store_true', help='generate relations only for sg_mode: 3')
    args = parser.parse_args()

    vg_eval = VGEval(args.model, args.bert, args.category,
                     debug=args.debug,
                     sg_mode=args.sg_mode,
                     region_mode=args.region_mode,
                     generate_relations=args.generate_relations
                    )
    results = vg_eval.eval(args.img, args.json, args.temperature)
    os.makedirs(Path(args.output).parent, exist_ok=True)
    torch.save(results, args.output)
    # pd.DataFrame(results).to_json(args.output, orient='records', lines=True)

