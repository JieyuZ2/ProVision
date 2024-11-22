from collections import Counter
from typing import List

import inflect

from ..attribute_classifier import AttributeClassifier
from ..template import get_qa_template
from ..utils import bbox_coordinate_to_ratio, check_object_for_counting_task, get_cnt_word, make_and_description
from ...base import BaseGenerator
from ...dataset import Attributes, JointDataset


class ExistsAttributeGenerator(BaseGenerator):
	"""
	"prompt"  : "How many {name} are there?",
	"response": "{count}"
	"""

	qa_templates = get_qa_template("ExistsAttributeGenerator")
	des_templates = [
		{
			"description": "There {be} {count_name}."
		}
	]
	inflect_engine = inflect.engine()
	metadata: bool = True

	def __init__(self, dataset: JointDataset, number_mode="random", **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.number_mode = number_mode

	def _generate(self, annotation: Attributes) -> List:
		data_list = []

		attributed_labels = []
		for attr, label in zip(annotation.attributes, annotation.labels):
			if len(attr):
				attributed_labels.append((tuple(sorted(attr)), label))

		for (attr, obj), cnt in Counter(attributed_labels).items():
			if cnt == 1:
				candidates = [1, 2, 3, 4]
			else:
				candidates = [cnt, cnt - 1, cnt + 1, cnt + 2]
			candidates = get_cnt_word(candidates, self.rng, self.inflect_engine, self.number_mode)
			cnt_word = candidates[0]

			be = 'is'
			attr_desc = make_and_description(attr, self.rng)
			name = f'{attr_desc} {obj}'

			if check_object_for_counting_task(obj):
				if cnt > 1:
					be = 'are'
					name = f'{attr_desc} {self.inflect_engine.plural(obj)}'
				count_name = cnt_word + ' ' + name
			else:
				if self.template_mode == 'qa':
					continue
				be = 'are'
				cnt_word = ''
				count_name = name

			data_list += self.make_one_data(
				{
					"name"      : name,
					"count_name": count_name,
					"be"        : be,
					"count"     : cnt_word,
					"candidates": candidates,
					"answer"    : cnt_word,
					"metadata"  : {
						"object"   : [obj],
						"attribute": attr,
						"count"    : [cnt]
					}
				},
			)
		return data_list


class AttributeBBoxGenerator(BaseGenerator):
	"""
	"prompt"  : "What are attributes of {name} at region {bbox}?",
	"response": "{attribute_values}"
	"""

	qa_templates = get_qa_template("AttributeBBoxGenerator")
	des_templates = [
		{
			"description": "The {name} at region {bbox} is {attribute_values}."
		}
	]
	attribute_classifier = AttributeClassifier()
	metadata: bool = True

	def _generate(self, annotation: Attributes) -> List:
		data_list = []
		height, width = annotation.height, annotation.width

		annotation = annotation.attributed_bboxes()

		for name, bbox, attributes in zip(annotation.labels, annotation.bboxes, annotation.attributes):
			candidates = self.attribute_classifier.sample_attribute(self.rng, n=self.n_choice - 1, exclude=attributes)
			answer = self.rng.choice(attributes)
			candidates = [answer] + candidates
			data_list += self.make_one_data(
				{
					"name"            : name,
					"bbox"            : str(bbox_coordinate_to_ratio(bbox, height, width)),
					"attribute_values": attributes,
					"candidates"      : candidates,
					"answer"          : answer,
					"metadata"        : {
						"object"   : [name],
						"attribute": attributes
					}
				},
			)

		return data_list


class TypedAttributeBBoxGenerator(BaseGenerator):
	"""
	"prompt"  : "What are {attribute_type} of {name} at region {bbox}?",
	"response": "{attribute_values}"
	"""

	qa_templates = get_qa_template("TypedAttributeBBoxGenerator")
	attribute_classifier = AttributeClassifier()
	metadata: bool = True

	def _generate(self, annotation: Attributes) -> List:
		data_list = []
		height, width = annotation.height, annotation.width

		annotation = annotation.attributed_bboxes()

		for name, bbox, attributes in zip(annotation.labels, annotation.bboxes, annotation.attributes):

			type_to_attributes = {}
			for attr in attributes:
				attribute_type = self.attribute_classifier.classify(attr)
				if attribute_type not in type_to_attributes:
					type_to_attributes[attribute_type] = []
				type_to_attributes[attribute_type].append(attr)

			for attribute_type, attr in type_to_attributes.items():
				if attribute_type != 'other':
					candidates = self.attribute_classifier.sample_attribute_from_category(attribute_type, self.rng, n=self.n_choice - 1, exclude=attr)
					answer = self.rng.choice(attr)
					candidates = [answer] + candidates
					data_list += self.make_one_data(
						{
							"name"            : name,
							"bbox"            : str(bbox_coordinate_to_ratio(bbox, height, width)),
							"attribute_values": attr,
							"attribute_type"  : attribute_type,
							"candidates"      : candidates,
							"answer"          : answer,
							"metadata"        : {
								"object"   : [name],
								"attribute": attr
							}
						},
					)

		return data_list


AttributeGeneratorList = [
	ExistsAttributeGenerator,
	AttributeBBoxGenerator,
	TypedAttributeBBoxGenerator
]
