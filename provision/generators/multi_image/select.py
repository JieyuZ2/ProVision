from collections import defaultdict
from typing import List

import numpy as np
from tqdm import tqdm

from ..template import get_qa_template
from ..utils import check_object_for_counting_task
from ...base import BaseMultiGenerator
from ...dataset import JointDataset


class HasObjectMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "which image has {object}?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("HasObjectMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		total_number = len(self.dataset.annotations)
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:
			selected = tuple(self.rng.choice(total_number, self.n_data, replace=False, shuffle=False))
			if selected not in samples:
				samples.add(selected)

				object_to_data = defaultdict(list)
				for i, di in enumerate(selected):
					annotation = self.dataset.annotations[di]
					for obj in set(annotation.labels):
						object_to_data[obj].append(i)

				candidate_objs = [obj for obj, d in object_to_data.items() if len(d) == 1]
				if len(candidate_objs) > 0:
					obj = self.rng.choice(candidate_objs)
					d = object_to_data[obj]
					answer = d[0]
					data_list += self.make_one_data(
						{
							"data_path": [self.dataset.data_paths[di] for di in selected],
							"object"   : obj,
							"answer_id": answer,
							"metadata" : {
								"object": [obj],
							}
						},
					)
					pbar.update(1)

		pbar.close()
		return data_list


class HasAttributedObjectMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "which image has {attribute} {object}?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("HasAttributedObjectMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:
			target_obj, selected = self.dataset.sample_data_and_obj(self.rng, self.n_data)
			if len(selected) == self.n_data:
				selected = tuple(sorted(selected))
				if (target_obj, selected) not in samples:
					samples.add((target_obj, selected))

					attribute_to_data = defaultdict(set)
					for i, di in enumerate(selected):
						annotation = self.dataset.annotations[di]
						for obj, attrs in zip(annotation.labels, annotation.attributes):
							if obj == target_obj:
								for attr in attrs:
									attribute_to_data[attr].add(i)

					candidate_attrs = [attr for attr, d in attribute_to_data.items() if len(d) == 1]
					if len(candidate_attrs) > 0:
						attribute = self.rng.choice(candidate_attrs)
						d = list(attribute_to_data[attribute])
						answer = d[0]
						data_list += self.make_one_data(
							{
								"data_path": [self.dataset.data_paths[di] for di in selected],
								"object"   : target_obj,
								"attribute": attribute,
								"answer_id": answer,
								"metadata" : {
									"object"   : [target_obj],
									"attribute": [attribute]
								}
							},
						)
						pbar.update(1)

		pbar.close()
		return data_list


class HasNotObjectMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "which image does not have  {object}?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("HasNotObjectMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:
			target_obj, selected = self.dataset.sample_data_and_obj(self.rng, self.n_data - 1)
			d = self.dataset.sample_data_without_obj(self.rng, target_obj, 1)
			selected = d + selected
			if len(selected) == self.n_data:
				selected = tuple(sorted(selected))
				if (target_obj, selected) not in samples:
					samples.add((target_obj, selected))

					answer = selected.index(d[0])
					data_list += self.make_one_data(
						{
							"data_path": [self.dataset.data_paths[di] for di in selected],
							"object"   : target_obj,
							"answer_id": answer,
							"metadata" : {
								"object": [target_obj],
							}
						},
					)
					pbar.update(1)

		pbar.close()
		return data_list


class HasNotAttributedObjectMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "which image does not have {attribute} {object}?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("HasNotAttributedObjectMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:
			target_obj, selected = self.dataset.sample_data_and_obj(self.rng, 2)
			selected = tuple(sorted(selected))
			if (target_obj, selected) not in samples:
				samples.add((target_obj, selected))

				attribute_to_data = defaultdict(set)
				for i, di in enumerate(selected):
					annotation = self.dataset.annotations[di]
					for obj, attrs in zip(annotation.labels, annotation.attributes):
						if obj == target_obj:
							for attr in attrs:
								attribute_to_data[attr].add(i)

				candidate_attrs = [attr for attr, d in attribute_to_data.items() if len(d) == 1]
				if len(candidate_attrs) > 0:
					attribute = self.rng.choice(candidate_attrs)
					d = list(attribute_to_data[attribute])
					answer = 1 - d[0]

					if self.n_data > 2:
						others = self.dataset.sample_data_without_obj(self.rng, target_obj, self.n_data - 2)
						selected = list(selected) + others

					data_list += self.make_one_data(
						{
							"data_path": [self.dataset.data_paths[di] for di in selected],
							"object"   : target_obj,
							"attribute": attribute,
							"answer_id": answer,
							"metadata" : {
								"object"   : [target_obj],
								"attribute": [attribute]
							}
						},
					)
					pbar.update(1)

		pbar.close()
		return data_list


class HasRelationMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "In which image {object1} is {relation} {object2}?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("HasRelationMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:

			objs, rel, d1 = self.dataset.sample_data_and_rel(self.rng, 1)
			d2 = self.dataset.sample_data_without_rel(self.rng, objs, rel, self.n_data - 1)
			selected = d1 + d2

			if len(selected) == self.n_data:
				selected = tuple(sorted(selected))
				if selected not in samples:
					samples.add(selected)

					answer = selected.index(d1[0])
					data_list += self.make_one_data(
						{
							"data_path": [self.dataset.data_paths[di] for di in selected],
							"relation" : rel,
							"object1"  : objs[0],
							"object2"  : objs[1],
							"answer_id": answer,
							"metadata" : {
								"relation": [rel],
								"object"  : list(objs),
							}
						},
					)
					pbar.update(1)

		pbar.close()
		return data_list


class HasNotRelationMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "In which image {object1} is not {relation} {object2}?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("HasNotRelationMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:

			objs, rel, d1 = self.dataset.sample_data_and_rel(self.rng, self.n_data - 1)
			d2 = self.dataset.sample_data_without_rel(self.rng, objs, rel, 1)
			selected = d1 + d2

			if len(selected) == self.n_data:
				selected = tuple(sorted(selected))
				if selected not in samples:
					samples.add(selected)

					answer = selected.index(d2[0])
					data_list += self.make_one_data(
						{
							"data_path": [self.dataset.data_paths[di] for di in selected],
							"relation" : rel,
							"object1"  : objs[0],
							"object2"  : objs[1],
							"answer_id": answer,
							"metadata" : {
								"relation": [rel],
								"object"  : list(objs),
							}
						},
					)
					pbar.update(1)

		pbar.close()
		return data_list


class HasMostObjectMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "which image has most {object}?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("HasMostObjectMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:
			res = self.dataset.sample_data_and_obj_diff_cnt(self.rng, self.n_data)
			if len(res):
				target_obj, l = res
				if check_object_for_counting_task(target_obj):
					selected, cnt = zip(*l)
					if len(selected) == self.n_data:
						selected_ = tuple(sorted(selected))
						if selected_ not in samples:
							samples.add(selected_)

							answer = int(np.argmax(cnt))
							data_list += self.make_one_data(
								{
									"data_path": [self.dataset.data_paths[di] for di in selected],
									"object"   : target_obj,
									"answer_id": answer,
									"metadata" : {
										"object": [target_obj],
									}
								},
							)
							pbar.update(1)

		pbar.close()
		return data_list


class HasLeastObjectMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "which image has least {object}?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("HasLeastObjectMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:
			res = self.dataset.sample_data_and_obj_diff_cnt(self.rng, self.n_data)
			if len(res):
				target_obj, l = res
				if check_object_for_counting_task(target_obj):
					selected, cnt = zip(*l)
					if len(selected) == self.n_data:
						selected_ = tuple(sorted(selected))
						if selected_ not in samples:
							samples.add(selected_)

							answer = int(np.argmin(cnt))
							data_list += self.make_one_data(
								{
									"data_path": [self.dataset.data_paths[di] for di in selected],
									"object"   : target_obj,
									"answer_id": answer,
									"metadata" : {
										"object": [target_obj],
									}
								},
							)
							pbar.update(1)

		pbar.close()
		return data_list


MultiSelectGeneratorList = [
	HasRelationMultiGenerator,
	HasNotRelationMultiGenerator,
	HasObjectMultiGenerator,
	HasNotObjectMultiGenerator,
	HasAttributedObjectMultiGenerator,
	HasNotAttributedObjectMultiGenerator,
	HasMostObjectMultiGenerator,
	HasLeastObjectMultiGenerator
]
