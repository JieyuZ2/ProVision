from collections import defaultdict
from typing import List

import inflect

from ..template import get_qa_template
from ..utils import bbox_coordinate_to_ratio, safe_sample
from ...base import BaseGenerator
from ...dataset import JointDataset, Relations


class RelationBaseGenerator(BaseGenerator):
	def __init__(self, dataset: JointDataset, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		relations = set()
		for annotation in dataset.annotations:
			for _, rel, _ in annotation.relations:
				relations.add(rel)
		self.relations = list(relations)

	def sample_relations(self, rng, n=1, exclude=[]):
		return safe_sample(rng, self.relations, n, exclude)


class ExistsRelationGenerator(RelationBaseGenerator):
	"""
	"prompt"  : "What is the relation between {object1} and {object2}?",
	"response": "{relation}"
	"""

	qa_templates = get_qa_template("ExistsRelationGenerator")
	des_templates = [
		{
			"description": "{object1} {be} {relation} {object2}."
		}
	]
	inflect_engine = inflect.engine()
	metadata: bool = True

	def _generate(self, annotation: Relations) -> List:
		data_list = []

		agg_relations = defaultdict(set)
		for o1, rel, o2 in annotation.relations:
			o1, o2 = annotation.labels[o1], annotation.labels[o2]
			if annotation.labels.count(o1) == 1 and annotation.labels.count(o2) == 1:
				agg_relations[(o1, o2)].add(rel)

		for (o1, o2), rel in agg_relations.items():
			rel = list(rel)
			be = 'are' if self.inflect_engine.singular_noun(o1) else 'is'

			candidates = self.sample_relations(self.rng, n=self.n_choice - 1, exclude=rel)
			answer = self.rng.choice(rel)
			candidates = [answer] + candidates

			data_list += self.make_one_data(
				{
					"object1"   : o1,
					"be"        : be,
					"relation"  : rel,
					"object2"   : o2,
					"candidates": candidates,
					"answer"    : answer,
					"metadata"  : {
						"object"  : [o1, o2],
						"relation": rel
					}
				},
			)

		return data_list


class RelationBBoxGenerator(RelationBaseGenerator):
	"""
	"prompt"  : "What is the relation between objects at {bbox1} and {bbox2}?",
	"response": "{relation}"
	"""

	qa_templates = get_qa_template("RelationBBoxGenerator")
	des_templates = [
		{
			"description": "Object at {bbox1} is {relation} object at {bbox2}."
		}
	]
	metadata: bool = True

	def _generate(self, annotation: Relations) -> List:
		data_list = []
		height, width = annotation.height, annotation.width

		agg_relations = defaultdict(list)
		for o1, rel, o2 in annotation.relations:
			agg_relations[(o1, o2)].append(rel)

		for (o1, o2), rel in agg_relations.items():
			candidates = self.sample_relations(self.rng, n=self.n_choice - 1, exclude=rel)
			answer = self.rng.choice(rel)
			candidates = [answer] + candidates

			data_list += self.make_one_data(
				{
					"bbox1"     : str(bbox_coordinate_to_ratio(annotation.bboxes[o1], height, width)),
					"relation"  : rel,
					"bbox2"     : str(bbox_coordinate_to_ratio(annotation.bboxes[o2], height, width)),
					"candidates": candidates,
					"answer"    : answer,
					"metadata"  : {
						"object"  : [annotation.labels[o1], annotation.labels[o2]],
						"relation": rel
					}
				},
			)

		return data_list


class HeadRelationGenerator(RelationBaseGenerator):
	"""
	"prompt"  : "Among {candidates}, what is {relation} {object2}?",
	"response": "{object1}"
	"""

	qa_templates = get_qa_template("HeadRelationGenerator")
	metadata: bool = True

	def _generate(self, annotation: Relations) -> List:
		data_list = []

		target_to_relations = defaultdict(dict)
		for o1, rel, o2 in annotation.relations:
			o1, o2 = annotation.labels[o1], annotation.labels[o2]
			if rel not in target_to_relations[o2]:
				target_to_relations[o2][rel] = set()
			target_to_relations[o2][rel].add(o1)

		all_objects = set(annotation.labels)

		for o2, rels in target_to_relations.items():
			for rel, o1s in rels.items():
				candidates = list(all_objects - o1s - {o2})
				if len(candidates) == 0:
					continue
				for o1 in o1s:
					if len(candidates) > self.n_choice - 1:
						sampled = [o1] + list(self.rng.choice(candidates, self.n_choice - 1, replace=False))
					else:
						sampled = [o1] + candidates
					data_list += self.make_one_data(
						{
							"object1"   : o1,
							"relation"  : rel,
							"object2"   : o2,
							"candidates": sampled,
							"answer"    : o1,
							"metadata"  : {
								"object"  : [o1, o2],
								"relation": [rel]
							}
						},
					)

		return data_list


RelationGeneratorList = [
	ExistsRelationGenerator,
	RelationBBoxGenerator,
	HeadRelationGenerator
]
