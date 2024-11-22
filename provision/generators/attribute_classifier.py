from .attribute_category import ATTRIBUTE_CATEGORY
from .utils import safe_sample


class AttributeClassifier:
	def __init__(self):
		attribute_category = ATTRIBUTE_CATEGORY
		self.categories = list(attribute_category.keys())
		self.attribute_to_category = {}
		self.attributes = []
		for category, attributes in attribute_category.items():
			self.attributes += attributes
			for attribute in attributes:
				self.attribute_to_category[attribute] = category
		self.category_to_attribute = attribute_category.copy()

	def classify(self, attribute):
		return self.attribute_to_category.get(attribute, "other")

	def sample_attribute(self, rng, n=1, exclude=[]):
		return safe_sample(rng, self.attributes, n, exclude)

	def sample_category(self, rng, n=1):
		return list(rng.choice(self.categories, n, replace=False))

	def sample_attribute_from_category(self, category, rng, n=1, exclude=[]):
		return safe_sample(rng, self.category_to_attribute[category], n, exclude)
