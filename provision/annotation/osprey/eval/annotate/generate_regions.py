import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from PIL import Image

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

from osprey.eval.utils import shard_data, is_hdf5_file, save_results_to_hdf5, check_processed_ids_from_hdf5

# set logging format
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def load_sam_mask_generator(sam_ckpt, sam_mode, device, output_mode="binary_mask"):
    assert output_mode in ["binary_mask", "coco_rle"], f"Invalid output_mode: {output_mode}"
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    from som.task_adapter.sam.tasks.automatic_mask_generator import load_sam_mask_generator, list_available_modes
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(device)
    print('Available sam modes:', list_available_modes())
    mask_generator = load_sam_mask_generator(sam_mode, sam, output_mode=output_mode)

    return mask_generator

def load_sam2_mask_generator(sam2_ckpt, device, output_mode="binary_mask"):
    assert output_mode in ["binary_mask", "coco_rle"], f"Invalid output_mode: {output_mode}"
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    model_cfg = "sam2_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_ckpt, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2, output_mode=output_mode)
    return mask_generator

class ImageDataset(Dataset):
    """
    {
        "image_id": "00453/004539375.jpg",
        "id": "004539375",
    }
    """
    def __init__(self, image_dir, image_file=None, 
                 num_shards=1, shard_index=0, identifier_key='id'):
        self.image_dir = image_dir
        self.identifier_key = identifier_key

        image_data = self.load_image_data(image_dir, image_file)
        image_data = [self.preprocess_data(data) for data in image_data]
        image_data = shard_data(image_data, num_shards, shard_index)
        self.image_data = {data["id"]: data for data in image_data}
        assert len(image_data) == len(self.image_data), "Duplicate identifiers found in image data"
        self.image_keys = list(self.image_data.keys())

        logging.info(f"Loaded {len(self.image_data)} images from {image_dir}")
    
    def load_image_data(self, image_dir, image_file) -> list[dict]:
        if image_file:
            with open(image_file) as f:
                image_data = json.load(f)
        else:
            image_data = []
            # Load all images in the directory
            for root, _, files in os.walk(image_dir):
                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')):
                        image_data.append({
                            'image_id': file,
                            'id': Path(file).stem
                        })
            image_data = image_data
        return image_data
    
    def preprocess_data(self, data: dict):
        return data
    
    def get_data(self):
        return self.image_data
    
    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx) -> dict:
        """
        {
            "image_id": "00453/004539375.jpg",
            "id": "004539375",
        }
        """
        
        if isinstance(idx, int):
            image_key = self.image_keys[idx]
            item: dict = self.load_image_item(image_key)
            return item
        elif isinstance(idx, slice):
            items = []
            for i in range(*idx.indices(len(self.image_keys))):
                image_key = self.image_keys[i]
                item: dict = self.load_image_item(image_key)
                items.append(item)
            return items
        else:
            raise TypeError("Invalid argument type.")

        # image_key = self.image_keys[idx]
        # item: dict = self.load_image_item(image_key)
        # return item
    
    def load_image_item(self, image_key):
        item = deepcopy(self.image_data[image_key])
        image_path = os.path.join(self.image_dir, item["image_id"])
        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image)
        item["image"] = image
        return item
    
    def filter_processed_ids(self, output):
        """ Filters out image data entries if their results already exist in the output directory """
        processed_ids: set = self._get_processed_ids(output)
        original_len = len(self.image_data)
        self.image_keys = [key for key in self.image_keys if key not in processed_ids]
        self.image_data = {key: self.image_data[key] for key in self.image_keys}
        logging.info(f"Filtered to {len(self.image_data)} images from {original_len} after excluding already processed ones.")
    
    def _get_processed_ids(self, output) -> set:
        if not os.path.exists(output):
            return set()
        if is_hdf5_file(output):
            return check_processed_ids_from_hdf5(output)
        if output.endswith('.jsonl'):
            with open(output) as f:
                return {json.loads(line)[self.identifier_key] for line in f}
        else:
            return {Path(f).stem for f in os.listdir(output)}

class LLaVAPretrainImageDataset(ImageDataset):
    def preprocess_data(self, data: dict): 
        data["image_id"] = data.pop("image")
        data["metadata/url"] = data.pop("url")
        data["metadata/blip_caption"] = data.pop("blip_caption").encode('ascii', 'ignore').decode('ascii')
        return data

class V3DetImageDataset(ImageDataset):
    def load_image_data(self, image_dir, image_file) -> list[dict]:
        with open(image_file) as f:
            annot = json.load(f)
            image_data = []
            # verify image files exist
            for data in annot['images']:
                image_id = data['file_name'].replace('images/', '')
                image_path = os.path.join(image_dir, image_id)
                data['id'] = str(data['id'])
                data['image_id'] = image_id
                if os.path.exists(image_path):
                    image_data.append(data)
                else:
                    print(f"Image file not found: {image_path}")
        return image_data

class VGImageDataset(ImageDataset):
    def load_image_data(self, image_dir, image_file) -> list[dict]:
        image_data = []
        with open(image_file) as f:
            annot = json.load(f)
        for data in tqdm(annot):
            image_data.append({
                'id': str(data['image_id']),
                'image_id': data['url'].split('/')[-1],
                'url': data['url'],
                'metadata/height': data['height'],
                'metadata/width': data['width'],
            }) 
        return image_data

class PSGDataset(ImageDataset):
    def load_image_data(self, image_dir, image_file) -> list[dict]:
        image_data = []
        with open(image_file) as f:
            annot = json.load(f)
        for data in tqdm(annot['data']):
            image_data.append({
                'id': str(data['index']),
                'image_id': data['file_name'],
                'question_id': data['question_id'],
                'metadata/height': data['height'],
                'metadata/width': data['width'],
                'metadata/annotations': data['annotations'],
            }) 
        return image_data

def load_dataset_class(dataset_name):
    if dataset_name == 'llava_pretrain':
        return LLaVAPretrainImageDataset
    elif dataset_name == 'v3_det':
        return V3DetImageDataset
    elif dataset_name == 'vg':
        return VGImageDataset
    elif dataset_name == 'psg':
        return PSGDataset
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

@dataclass
class RegionProposalResult:
    id: str
    image_id: str
    regions: list[dict]
    
class RegionProposalPipeline:
    def __init__(self, mask_generator):
        self.mask_generator = mask_generator
    
    def _generate_region_results(self, datum, regions) -> dict:
        """ Returns results to save"""
        result = {
            'id': datum["id"],
            'image_id': datum["image_id"],
            'regions': regions
        }
        for k in datum.keys():
            if k.startswith("metadata"):
                result[k] = datum[k]
        
        return result
    
    def __call__(self, image_dataset: dict | list[dict] | ImageDataset, 
                 disable_tqdm: bool = False
        ) -> list[dict]:
        results = []

        if isinstance(image_dataset, dict):
            image_dataset = [image_dataset]

        for datum in tqdm(image_dataset, disable=disable_tqdm):
            image: np.ndarray = datum["image"]
            regions = self.mask_generator.generate(image)
            result = self._generate_region_results(datum, regions)
            results.append(result)

        return results

    def process_and_save_batches(self, data_loader: DataLoader, output, disable_tqdm: bool = False) -> list[dict]:
        for batches in tqdm(data_loader, disable=disable_tqdm):
            results: list[dict] = self.__call__(batches, disable_tqdm=disable_tqdm)
            self.save_results(results, output)
            logging.info(f"Saved {len(results)} regions to {output}")

            avg_regions = np.mean([len(r['regions']) for r in results])
            logging.info(f"Average number of regions per image: {avg_regions}")
    
    def save_results(self, results: list[dict], output: str):
        if is_hdf5_file(output):
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            save_results_to_hdf5(results, output)
        else:
            Path(output).mkdir(parents=True, exist_ok=True)
            for result in results:
                with open(os.path.join(output, f"{result['id']}.json"), "w") as f:
                    json.dump(result, f)


if __name__ == '__main__':
    '''
    python -m osprey.eval.annotate.generate_regions \
        --dataset_name llava_pretrain \
        --image_data /net/nfs.cirrascale/mosaic/jamesp/images/llava/LLaVA-Pretrain/blip_laion_cc_sbu_558k_meta.json \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/images/llava/LLaVA-Pretrain/images \
        --sam_mode whole \
        --num_shards 16 \
        --shard_index 0 \
        --output region_results/llava-pretrain/sam_whole_16_0.hdf5

    # sam-2?
    python -m osprey.eval.annotate.generate_regions \
        --dataset_name llava_pretrain \
        --image_data /net/nfs.cirrascale/mosaic/jamesp/images/llava/LLaVA-Pretrain/blip_laion_cc_sbu_558k_meta.json \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/images/llava/LLaVA-Pretrain/images \
        --mask_generator_path /net/nfs.cirrascale/mosaic/jamesp/models/segment-anything/sam2_hiera_large.pt \
        --mask_generator_model sam2 \
        --output region_results/llava-pretrain/sam2.hdf5
    
    # vg test
    python -m osprey.eval.annotate.generate_regions \
        --dataset_name vg \
        --image_data /net/nfs.cirrascale/mosaic/jamesp/data/vg/test_image_data.json \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/images/vg/VG_100K_all \
        --sam_mode whole \
        --output region_results/vg_test/sam_whole.hdf5
    
    # psg test
    python -m osprey.eval.annotate.generate_regions \
        --dataset_name psg \
        --image_data /net/nfs.cirrascale/mosaic/jamesp/Osprey/osprey/eval/psg/psg_asv2_val_test.json \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/images/coco/ \
        --sam_mode whole \
        --output region_results/psg_test/sam_whole.hdf5   
         
    # v3_det
    python -m osprey.eval.annotate.generate_regions \
        --dataset_name v3_det \
        --image_data /net/nfs.cirrascale/mosaic/jamesp/data/V3Det/annotations/v3det_2023_v1_train.json \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/data/V3Det/images \
        --sam_mode whole \
        --output region_results/v3_det/sam_whole.hdf5
    
    # ade20k
    python -m osprey.eval.annotate.generate_regions \
        --dataset_name ade_20k \
        --image_data /net/nfs.cirrascale/mosaic/jamesp/data/V3Det/annotations/v3det_2023_v1_train.json \
        --image_dir /net/nfs.cirrascale/mosaic/jamesp/data/V3Det/images \
        --sam_mode whole \
        --output region_results/v3_det/sam_whole.hdf5
    
    '''
    parser = argparse.ArgumentParser(description='osprey sg generator', formatter_class=argparse.RawTextHelpFormatter)
    # Model config
    parser.add_argument('--dataset_name', help='image json file', required=True,) 
    parser.add_argument('--image_data', help='image json file', required=True) 
    parser.add_argument('--image_dir', help='path to images', required=True)
    parser.add_argument('--mask_generator_path', help='path to mask_generator', 
                        default='/net/nfs.cirrascale/mosaic/jamesp/models/segment-anything/sam_vit_h_4b8939.pth')
    parser.add_argument('--mask_generator_model', help='mask_generator model to use', default='sam', choices=['sam', 'sam2'])
    parser.add_argument('--sam_mode', help='custom mask generator mode to use for sam model', default='whole', choices=['whole', 'default', 'part'])
    parser.add_argument('--output', help='hdf5 or directory to save results', default=None)

    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards to split the data into")
    parser.add_argument("--shard_index", type=int, default=0, help="Index of the shard to process")
    parser.add_argument("--overwrite_output_file", action="store_true", help="Overwrite the output file if it exists")

    # Dataloader
    parser.add_argument('--num_workers', help='num_workers', default=4, type=int)
    parser.add_argument('--batch_size', help='batch size', default=100, type=int)

    args = parser.parse_args()

    # Load data
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    dataset_class = load_dataset_class(args.dataset_name)
    dataset: ImageDataset = dataset_class(image_dir=args.image_dir, image_file=args.image_data,
                                        num_shards=args.num_shards, shard_index=args.shard_index,)
    dataset.filter_processed_ids(args.output)
    data_loader = DataLoader(dataset, 
                             args.batch_size, 
                             shuffle=False, 
                             drop_last=False, 
                             num_workers=args.num_workers,
                             collate_fn=lambda x: x
                            )
    logging.info(f"Number of batches: {len(data_loader)} with batch size {args.batch_size}")

    # Load region proposal model
    if args.mask_generator_model == 'sam2':
        mask_generator = load_sam2_mask_generator(args.mask_generator_path, device='cuda', output_mode='coco_rle')
    else:
        mask_generator = load_sam_mask_generator(args.mask_generator_path, sam_mode=args.sam_mode, device='cuda', output_mode='coco_rle')

    # generate scene graph with regions
    region_proposal = RegionProposalPipeline(mask_generator)
    region_proposal.process_and_save_batches(data_loader, output=args.output)





    
    