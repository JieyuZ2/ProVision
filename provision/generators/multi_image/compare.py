from collections import defaultdict
from itertools import permutations
from typing import List
from itertools import chain
from tqdm import tqdm

from ..utils import make_and_description
from ...base import BaseMultiGenerator
from ...dataset import JointDataset
from ..template import get_qa_template


def check_difference(l):
	l = [tuple(sorted(i)) for i in l]
	return len(set(l)) == len(l)


def describe_attributes(obj, attributes):
	desc = f"{obj} is "
	for i, attr in enumerate(attributes):
		desc += f"{make_and_description(attr)} in Image {i}, "
	return desc[:-2] + "."


class CompareAttributeMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "What is the difference of attributes of {object} in these images?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("CompareAttributeMultiGenerator")
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

					data_to_attribute = defaultdict(set)
					for i, di in enumerate(selected):
						annotation = self.dataset.annotations[di]
						labels = annotation.labels
						if labels.count(target_obj) > 1:
							break
						for obj, attrs in zip(annotation.labels, annotation.attributes):
							if obj == target_obj and len(attrs) > 0:
								data_to_attribute[i].update(attrs)

					if len(data_to_attribute) == self.n_data and check_difference(data_to_attribute.values()):
						attributes = tuple(data_to_attribute.values())

						answer = describe_attributes(target_obj, attributes)

						perm = [i for i in permutations(attributes) if i != attributes]
						if len(perm) > self.n_choice - 1:
							perm = self.rng.choice(perm, self.n_choice - 1, replace=False).tolist()
						candidates = [answer] + [describe_attributes(target_obj, p) for p in perm]

						attributes = list(chain(*attributes))
						data_list += self.make_one_data(
							{
								"data_path" : [self.dataset.data_paths[di] for di in selected],
								"object"    : target_obj,
								"attribute" : attributes,
								"candidates": candidates,
								"answer"    : answer,
								"metadata"  : {
									"object"   : [target_obj],
									"attribute": attributes
								}
							},
						)
						pbar.update(1)

		pbar.close()
		return data_list


def describe_relations(obj1, obj2, relations):
	desc = f"{obj1} is "
	for i, rel in enumerate(relations):
		desc += f"{make_and_description(rel)} {obj2} in Image {i}, "
	return desc[:-2] + "."


class CompareRelationMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "What is the difference of the relation between {object1} and {object2} in these images?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("CompareRelationMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:
			obj1, obj2, selected = self.dataset.sample_data_and_object_pair(self.rng, self.n_data)
			if len(selected) == self.n_data:
				selected = tuple(sorted(selected))
				if (obj1, obj2, selected) not in samples:
					samples.add((obj1, obj2, selected))

					data_to_relation = defaultdict(set)
					for i, di in enumerate(selected):
						annotation = self.dataset.annotations[di]
						labels = annotation.labels
						if labels.count(obj1) > 1 or labels.count(obj2) > 1:
							break

						for o1, rel, o2 in annotation.relations:
							if labels[o1] == obj1 and labels[o2] == obj2:
								data_to_relation[i].add(rel)

					if len(data_to_relation) == self.n_data and check_difference(data_to_relation.values()):
						relations = tuple(data_to_relation.values())

						answer = describe_relations(obj1, obj2, relations)

						perm = [i for i in permutations(relations) if i != relations]
						if len(perm) > self.n_choice - 1:
							perm = self.rng.choice(perm, self.n_choice - 1, replace=False).tolist()
						candidates = [answer] + [describe_relations(obj1, obj2, p) for p in perm]

						relations = list(chain(*relations))
						data_list += self.make_one_data(
							{
								"data_path" : [self.dataset.data_paths[di] for di in selected],
								"object1"   : obj1,
								"object2"   : obj2,
								"relation"  : relations,
								"candidates": candidates,
								"answer"    : answer,
								"metadata"  : {
									"object"  : [obj1, obj2],
									"relation": relations
								}
							},
						)
						pbar.update(1)

		pbar.close()
		return data_list


MultiCompareGeneratorList = [
	CompareRelationMultiGenerator,
	CompareAttributeMultiGenerator,
]
