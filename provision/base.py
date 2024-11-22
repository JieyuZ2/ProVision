import json
import os
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from itertools import chain
from multiprocessing import Pool
from typing import Callable, Dict, List, Union

import numpy as np
from tqdm import tqdm

from .generators.utils import make_one_data

metadata_fields = ['object', 'attribute', 'relation', 'count']


def get_empty_metadata_count():
	return {
		'object'   : {},
		'attribute': {},
		'relation' : {},
		'count'    : {},
	}


class BaseDataset:
	data_paths: List[str]
	raw_annotations: List
	sources: List[str]
	annotations: List

	def __init__(self, annotation_path: Union[str, List[str]]):
		self.annotation_path = annotation_path

		if isinstance(annotation_path, str):
			with open(annotation_path, 'r') as f:
				annotation = json.load(f)
				self.data_paths = []
				self.raw_annotations = []
				self.sources = []
				for ann in annotation.values():
					self.data_paths.append(ann['data_path'])
					self.raw_annotations.append(ann['annotation'])
					self.sources.append(ann.get('source', None))

		else:
			pre_annotation = json.load(open(self.annotation_path[0], 'r'))  # key: image_id (data_path), value: {data_path, annotation, ...}
			for path in annotation_path[1:]:
				with open(path, 'r') as f:
					annotation = json.load(f)
					for key, ann in annotation.items():
						pre_annotation[key]['annotation'].update(ann['annotation'])

			self.data_paths = []
			self.raw_annotations = []
			self.sources = []
			for ann in pre_annotation.values():
				self.data_paths.append(ann['data_path'])
				self.raw_annotations.append(ann['annotation'])
				self.sources.append(ann.get('source', None))

		self._load()

	@abstractmethod
	def _load(self):
		"""
		(Abstract method) load"
		"""

	def __getitem__(self, idx):
		return self.annotations[idx]

	def __len__(self):
		return len(self.data_paths)

	def subset(self, indices):
		new_dataset = deepcopy(self)
		new_dataset.data_paths = [self.data_paths[i] for i in indices]
		new_dataset.raw_annotations = [self.raw_annotations[i] for i in indices]
		new_dataset.sources = [self.sources[i] for i in indices]
		new_dataset._load()
		return new_dataset


class BaseGenerator:
	dataset: BaseDataset
	qa_templates = []
	des_templates = []

	def __init__(
			self,
			dataset: BaseDataset,
			template_mode: str = 'qa',
			return_templated=False,
			enumerate_templates=False,
			multi_choice_ratio: float = 0.0,  # effective only when template_mode == 'qa'
			n_choice: int = 4,  # effective only when template_mode == 'qa'
			n_sample_per_generator: int = 1,  # effective only when template_mode == 'qa'
			seed: int = 42,
			cache_file: str = None,
			**kwargs
	):
		"""
		templates = 'qa' or 'description'
		"""
		self.dataset = dataset
		self.return_templated = return_templated
		self.enumerate_templates = enumerate_templates
		self.multi_choice_ratio = multi_choice_ratio
		self.n_choice = n_choice
		self.n_sample_per_generator = n_sample_per_generator
		self.cache_file = cache_file


		if template_mode == 'qa':
			self.templates = self.qa_templates
			self.template_mode = 'qa'
		elif template_mode == 'description':
			self.templates = self.des_templates
			self.template_mode = 'description'
		else:
			raise ValueError(f"Invalid template mode: {template_mode}")
		self.rng = np.random.default_rng(seed)

	def _sample(self, datat_list, indices, metadata_count):
		p = []
		for i in indices:
			data = datat_list[i]
			cnt = []
			for k, vs in data['metadata'].items():
				for v in vs:
					# cnt.append(np.exp(metadata_count[k].get(v, 1)))
					cnt.append(metadata_count[k].get(v, 1) ** 3)
			p.append(1 / np.mean(cnt))
		p = np.array(p) / np.sum(p)
		c = list(self.rng.choice(indices, self.n_sample_per_generator, replace=False, p=p))
		return c

	def make_one_data(self, source, force=False, multiple_choice_ratio=None):
		if self.return_templated or force:
			if multiple_choice_ratio is None:
				multiple_choice_ratio = self.multi_choice_ratio
			if self.rng.random() < multiple_choice_ratio:
				return make_one_data(source, self.templates, self.rng, self.enumerate_templates, multiple_choice=True)
			else:
				return make_one_data(source, self.templates, self.rng, self.enumerate_templates, multiple_choice=False)
		else:
			return [source]

	def collect_metadata_helper(self, annotation):
		metadata_count = get_empty_metadata_count()
		for data in self._generate(annotation):
			for k, v in data['metadata'].items():
				for vv in v:
					if vv not in metadata_count[k]:
						metadata_count[k][vv] = 0
					metadata_count[k][vv] += 1
		return metadata_count

	def collect_metadata(self, n_workers=10, n_sample=1e4) -> Dict:
		if len(self.templates) == 0:
			return {}

		assert n_sample > 0, "n_sample should be greater than 0"
		if n_sample < len(self.dataset.annotations):
			annotations = list(self.rng.choice(self.dataset.annotations, int(n_sample), replace=False))
		else:
			annotations = self.dataset.annotations

		metadata_count = get_empty_metadata_count()

		if n_workers > 1:

			worker = partial(
				self.collect_metadata_helper,
			)

			with Pool(n_workers) as p:

				for metadata_count_i in p.map(worker, annotations):
					for k, v in metadata_count_i.items():
						for vv, cnt in v.items():
							if vv not in metadata_count[k]:
								metadata_count[k][vv] = 0
							metadata_count[k][vv] += cnt

		else:

			for annotation in (
					tqdm(
						annotations,
						desc=self.__class__.__name__,
						total=len(self.dataset.annotations),
					)
			):
				metadata_count_i = self.collect_metadata_helper(annotation)
				for k, v in metadata_count_i.items():
					for vv, cnt in v.items():
						if vv not in metadata_count[k]:
							metadata_count[k][vv] = 0
						metadata_count[k][vv] += cnt

		return metadata_count

	def generate_helper(self, source, metadata_count=None):
		data_path, annotation, source = source
		data_list = []

		if self.cache_file is not None:
			name = f"{self.__class__.__name__}:{data_path}"
			if name in self.cache:
				# print(f"cache hit: {name}")
				return self.cache[name]
			# else:
			# 	print(f"cache miss: {name}")

		if metadata_count is None:
			for data in self._generate(annotation):
				data['data_path'] = data_path
				data['generator'] = self.__class__.__name__
				data_list.append(data)
		else:
			candidates = self._generate(annotation)
			if len(candidates) == 0:
				if self.cache_file is not None:
					self.cache[name] = []
				return []
			indices = list(range(len(candidates)))
			if len(indices) <= self.n_sample_per_generator:
				sampled = indices
			else:
				if len(candidates[0]['metadata']):
					sampled = self._sample(candidates, indices, metadata_count)
				else:
					sampled = list(self.rng.choice(indices, self.n_sample_per_generator, replace=False))
			for i in sampled:
				data = candidates[i]
				data['data_path'] = data_path
				data['generator'] = self.__class__.__name__
				data_list.append(data)

		if self.cache_file is not None:
			self.cache[name] = data_list
			if (len(self.cache)+1) % 1000 == 0:
				json.dump(self.cache, open(self.cache_file, 'w'), indent=2)

		return data_list

	def generate(self, metadata_count=None, verbose=True, n_workers=10) -> List:
		if len(self.templates) == 0:
			return []

		if self.cache_file is not None:
			if not os.path.exists(self.cache_file):
				self.cache = {}
			else:
				self.cache = json.load(open(self.cache_file))
				print(f"cache loaded: {self.cache_file}")
				print(f"cache size: {len(self.cache)}")

		if n_workers > 1:

			worker = partial(
				self.generate_helper,
				metadata_count=metadata_count,
			)

			with Pool(n_workers) as p:
				data_list = p.map(worker, zip(self.dataset.data_paths, self.dataset.annotations, self.dataset.sources))
				data_list = list(chain(*data_list))
		else:
			data_list = []
			for data_path, annotation, source in (
					tqdm(
						zip(self.dataset.data_paths, self.dataset.annotations, self.dataset.sources),
						desc=self.__class__.__name__,
						total=len(self.dataset.data_paths),
						disable=not verbose
					)
			):
				data_list += self.generate_helper((data_path, annotation, source), metadata_count)

		if self.cache_file is not None:
			json.dump(self.cache, open(self.cache_file, 'w'), indent=2)

		return data_list

	@abstractmethod
	def _generate(self, annotation) -> List[Dict]:
		"""
		Abstract method
		"""


class BaseMultiGenerator(BaseGenerator):

	def __init__(
			self,
			dataset: BaseDataset,
			n_sample: int = 100,
			n_data: int = 2,
			**kwargs
	):
		super().__init__(
			dataset=dataset,
			**kwargs
		)
		self.n_sample = n_sample
		self.n_data = n_data
		self.candidates = [f'Image {i}' for i in range(self.n_data)]

	def make_one_data(self, source, force=False, multiple_choice_ratio=None):
		assert self.n_data == len(set(source['data_path'])), "The number of candidates should be equal to n_data"

		if self.return_templated or force:
			if multiple_choice_ratio is None:
				multiple_choice_ratio = self.multi_choice_ratio

			if "answer_id" in source:
				answer = source['data_path'][source['answer_id']]
				data_path = list(self.rng.permutation(source['data_path']))
				source['data_path'] = data_path
				source['candidates'] = self.candidates.copy()
				source['answer'] = source['candidates'][data_path.index(answer)]

			data_list = []
			for data in make_one_data(source, self.templates, self.rng, self.enumerate_templates, multiple_choice=self.rng.random() < multiple_choice_ratio):
				data['data_path'] = source['data_path']
				data_list.append(data)
			return data_list
		else:
			return [source]

	def generate(self, **kwargs) -> List:
		if len(self.templates) == 0:
			return []

		data_list = []
		for data in self._generate(self.n_sample):
			data['generator'] = self.__class__.__name__
			data_list.append(data)

		return data_list

	@abstractmethod
	def _generate(self, n_sample: int) -> List[Dict]:
		"""
		Abstract method
		"""


class JointGenerator(BaseGenerator):
	def __init__(self, dataset: BaseDataset, generators: List[Callable], **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.generators = [
			generator(dataset=dataset, seed=self.rng.integers(0, 1e4), **kwargs) for generator in generators
		]
		self.name_to_generator = {g.__class__.__name__: g for g in self.generators}

	def template(self, data_list, multiple_choice_ratio=None):
		# LLAVA format
		data = []
		for d in tqdm(data_list):
			generator: BaseGenerator = self.name_to_generator[d['generator']]
			for di in generator.make_one_data(d, force=True, multiple_choice_ratio=multiple_choice_ratio):
				if self.rng.choice([True, False]):
					di['prompt'] = di['prompt'] + '\n<image>'
				else:
					di['prompt'] = '<image>\n' + di['prompt']
				data.append(
					{
						"id"           : len(data),
						"image"        : d['data_path'],
						"conversations": [
							{
								"from" : "human",
								"value": di['prompt']
							},
							{
								"from" : "gpt",
								"value": di['response']
							}
						]
					}
				)
		return data

	def multi_image_template(self, data_list, multiple_choice_ratio=None):
		# LLAVA format
		data = []
		for d in tqdm(data_list):
			generator: BaseGenerator = self.name_to_generator[d['generator']]
			for di in generator.make_one_data(d, force=True, multiple_choice_ratio=multiple_choice_ratio):
				image_str = ['<image>'] * len(d['data_path'])
				image_str = ' '.join(image_str)
				if self.rng.choice([True, False]):
					di['prompt'] = di['prompt'] + f'\n{image_str}'
				else:
					di['prompt'] = f'{image_str}\n' + di['prompt']
				data.append(
					{
						"id"           : len(data),
						"image"        : d['data_path'],
						"conversations": [
							{
								"from" : "human",
								"value": di['prompt']
							},
							{
								"from" : "gpt",
								"value": di['response']
							}
						]
					}
				)
		return data

	def generate(self, metadata_count=None, verbose=True, n_workers=10) -> List:
		data_list = []
		for generator in tqdm(self.generators, desc='generating'):
			data_list += generator.generate(metadata_count=metadata_count, verbose=verbose, n_workers=n_workers)
		return data_list

	def collect_metadata(self, n_workers=10, n_sample=1e4):
		metadata_count = get_empty_metadata_count()
		for generator in tqdm(self.generators, desc='collecting metadata'):
			if generator.metadata:
				metadata_count_i = generator.collect_metadata(n_workers=n_workers, n_sample=n_sample)
				for k, v in metadata_count_i.items():
					for vv, cnt in v.items():
						if vv not in metadata_count[k]:
							metadata_count[k][vv] = 0
						metadata_count[k][vv] += cnt
		return metadata_count

	def _generate(self, annotation) -> List[Dict]:
		pass
