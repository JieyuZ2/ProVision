import functools
import json
from itertools import combinations

import numpy as np


def scene_graph_adjacent_objects(scene_graph, node):
	adjacent_objects = {}
	for edge in scene_graph["objects"][node]['relations']:
		obj = edge['object']
		if obj not in adjacent_objects:
			adjacent_objects[obj] = []
		adjacent_objects[obj].append((edge['name'], 0))

	for obj, edges in scene_graph["objects"].items():
		for edge in edges["relations"]:
			if edge['object'] == node:
				if obj not in adjacent_objects:
					adjacent_objects[obj] = []
				adjacent_objects[obj].append((edge['name'], 1))
	return adjacent_objects


def subgraph_to_json_str(subgraph, scene_graph):
	subgraph_json = {
		"attributes"      : [],
		"adjacent_objects": [],
	}
	adjacent_object_info = {}
	for element in subgraph:
		if isinstance(element, str):
			subgraph_json["attributes"].append(element)
		else:
			if len(element) == 2:
				obj, attr = element
				if obj not in adjacent_object_info:
					adjacent_object_info[obj] = {
						"attributes": [attr],
						"relation"  : None
					}
				else:
					adjacent_object_info[obj]["attributes"].append(attr)
			else:
				obj, rel, direction = element
				if obj not in adjacent_object_info:
					adjacent_object_info[obj] = {
						"attributes": [],
						"relation"  : (rel, direction)
					}
				else:
					adjacent_object_info[obj]["relation"] = (rel, direction)

	for obj, info in adjacent_object_info.items():
		subgraph_json["adjacent_objects"].append({
			"object"    : scene_graph["objects"][obj]["name"],
			"attributes": sorted(info["attributes"]),
			"relation"  : info["relation"]
		})
	subgraph_json["attributes"] = sorted(subgraph_json["attributes"])
	subgraph_json["adjacent_objects"] = sorted(subgraph_json["adjacent_objects"], key=lambda x: json.dumps(x))
	return json.dumps(subgraph_json)


def constrained_combinations(n, k, constraints):
	"""
	Generate all combinations of k elements from n elements that satisfy the constraints
	:param n:
	:param k:
	:param constraints: a list of tuples (i, j) that means that when i is not selected, i+1 ~ j should not be selected
	:return: a binary array of shape (x, n) where each row represents a valid combination
	"""
	combo = np.array(list(combinations(range(n), k)))
	selection = np.zeros((len(combo), n), dtype=bool)
	selection[np.arange(len(combo))[:, None], combo] = 1
	for start, end in constraints:
		selection = selection[~((selection[:, start] == 0) & (np.any(selection[:, start + 1:end], axis=1)))]
	return selection


def compose_parallel_phrase(phrases):
	if len(phrases) == 0:
		return ""
	elif len(phrases) == 1:
		return phrases[0]
	elif len(phrases) == 2:
		return f"{phrases[0]} and {phrases[1]}"
	else:
		phrases[-1] = "and " + phrases[-1]
		return ", ".join(phrases)


def compose_attributed_name(attributes, name):
	if len(attributes) > 0:
		attributes = compose_parallel_phrase(attributes)
		return f"{attributes} {name}"
	else:
		return name


@functools.lru_cache(maxsize=100000)
def compose_object_reference(subgraph: str):
	subgraph = json.loads(subgraph)

	# Helper function to create relation phrases
	def create_relation_phrase(attributed_name, relation_name, is_forward=True):
		return f"is {relation_name} the {attributed_name}" if is_forward else f"the {attributed_name} is {relation_name}"

	# Process relations
	forward_relations, backward_relations = [], []

	for idx, node in enumerate(subgraph['adjacent_objects']):
		rel = node['relation']
		attributed_name = compose_attributed_name(node.get("attributes", []), node['object'])
		if rel[1] == 0:
			forward_relations.append(create_relation_phrase(attributed_name, rel[0], True))
		else:
			backward_relations.append(create_relation_phrase(attributed_name, rel[0], False))

	# Combine relations into reference string
	reference = ""
	if forward_relations:
		reference += compose_parallel_phrase(forward_relations)
	if backward_relations:
		if forward_relations:
			reference += ", and also, "
		reference += compose_parallel_phrase(backward_relations)
	return reference


def subgraph_contain_multiple_same_direction_relations(subgraph):
	out_rel = False
	in_rel = False
	for item in subgraph:
		if len(item) == 3:
			if item[2] == 0:
				if out_rel:
					return True
				out_rel = True
			else:
				if in_rel:
					return True
				in_rel = True
	return False


def subgraph_contain_multiple_relations(subgraph):
	rel = False
	for item in subgraph:
		if isinstance(item, tuple) and len(item) == 3:
			if rel:
				return True
			rel = True
	return False


from ..attribute_classifier import AttributeClassifier

attribute_classifier = AttributeClassifier()


def _enumerate_subgraphs_w_object(
		scene_graph,
		start_node,
		subgraph_size=3,
		exclude_attribute_type=None,
		exclude_object=[]
):
	stamp = []
	elements = [
		attr for attr in scene_graph["objects"][start_node]['attributes']
		if exclude_attribute_type is None or attribute_classifier.classify(attr) != exclude_attribute_type
	]
	adjacent_objects = scene_graph_adjacent_objects(scene_graph, start_node)
	for obj in adjacent_objects:
		if obj not in exclude_object:
			start = len(elements)
			elements.append(obj)
			elements += [(obj, attr) for attr in scene_graph["objects"][obj]['attributes']]
			stamp.append((start, len(elements)))
	if len(elements) < subgraph_size:
		return []

	# sample all subgraphs that contain the start node with the given size
	selection = constrained_combinations(len(elements), subgraph_size, stamp)

	# distinguish subgraphs with and without the objects
	with_object_mask = np.any(selection[:, [start for start, _ in stamp]], axis=1)
	subgraphs_w_objects = [[elements[i] for i in np.where(indices)[0]] for indices in selection[with_object_mask]]
	subgraphs_wo_objects = [[elements[i] for i in np.where(indices)[0]] for indices in selection[~with_object_mask]]

	# for subgraph with object, add its all possible relations to the start node
	for obj, rels in adjacent_objects.items():
		new_subgraphs = []
		for subgraph in subgraphs_w_objects:
			if obj in subgraph:
				obj_id = subgraph.index(obj)
				for rel, direction in rels:
					subgraph_rel = subgraph.copy()
					subgraph_rel[obj_id] = (obj, rel, direction)
					# remove subgraphs with multiple same-direction relations
					if not subgraph_contain_multiple_relations(subgraph_rel):
						new_subgraphs.append(subgraph_rel)
			else:
				new_subgraphs.append(subgraph)
		subgraphs_w_objects = new_subgraphs

	subgraph_json_strs = [subgraph_to_json_str(subgraph, scene_graph)
						  for subgraph in subgraphs_w_objects + subgraphs_wo_objects]

	return set(subgraph_json_strs)


def enumerate_object_subgraphs(scene_graph, subgraph_size=3):
	subgraph_to_objects = {}
	for object, info in scene_graph["objects"].items():
		obj_name = info['name']
		subgraphs = _enumerate_subgraphs_w_object(scene_graph, object, subgraph_size)
		for subgraph in subgraphs:
			if subgraph not in subgraph_to_objects:
				subgraph_to_objects[subgraph] = set()
			subgraph_to_objects[subgraph].add(obj_name)
	return subgraph_to_objects


def generate_object_qa(scene_graph, subgraph_size=3):
	subgraph_to_nodes = enumerate_object_subgraphs(scene_graph, subgraph_size=subgraph_size)
	qas = []
	for subgraph_str, answers in subgraph_to_nodes.items():

		obj_reference = compose_object_reference(subgraph_str)
		subgraph = json.loads(subgraph_str)

		attributed_name = compose_attributed_name(subgraph.get("attributes", []), "object")

		if obj_reference != "":
			obj_reference = f" that {obj_reference}"
		question = f"What is the {attributed_name}{obj_reference}?"

		qas.append((question, list(answers)))

	return qas


def enumerate_attribute_subgraphs(scene_graph, subgraph_size=4):
	subgraph_to_nodes = {}
	for node, info in scene_graph["objects"].items():
		for attr in info['attributes']:
			attr_type = attribute_classifier.classify(attr)
			subgraphs = _enumerate_subgraphs_w_object(scene_graph, node, subgraph_size, exclude_attribute_type=attr_type)
			for subgraph in subgraphs:
				if subgraph not in subgraph_to_nodes:
					subgraph_to_nodes[subgraph] = {}
				if attr_type not in subgraph_to_nodes[subgraph]:
					subgraph_to_nodes[subgraph][attr_type] = set()
				subgraph_to_nodes[subgraph][attr_type].add(attr)
	return subgraph_to_nodes


def generate_attribute_qa(scene_graph, subgraph_size=3):
	subgraphs_to_attrs = enumerate_attribute_subgraphs(scene_graph, subgraph_size=subgraph_size)
	qas = []
	for subgraph_str, attributes in subgraphs_to_attrs.items():
		for attribute_type, attribute_set in attributes.items():

			obj_reference = compose_object_reference(subgraph_str)
			subgraph = json.loads(subgraph_str)

			attributed_name = compose_attributed_name(subgraph.get("attributes", []), "object")

			if obj_reference != "":
				obj_reference = f" that {obj_reference}"
			attribute_type_word = lambda x: "attribute" if x == "other" else x
			question = f"What is the {attribute_type_word(attribute_type)} of the {attributed_name}{obj_reference}?"

			qas.append((question, attribute_type, list(attribute_set)))

	return qas


def enumerate_relation_subgraphs(scene_graph, subgraph_size=4):
	subgraph_to_nodes_cnt = {}
	for node, info in scene_graph["objects"].items():
		subgraphs = _enumerate_subgraphs_w_object(scene_graph, node, subgraph_size)
		for subgraph in subgraphs:
			if subgraph not in subgraph_to_nodes_cnt:
				subgraph_to_nodes_cnt[subgraph] = 0
			subgraph_to_nodes_cnt[subgraph] += 1

	relations = {}
	for node, info in scene_graph["objects"].items():
		for rel in info['relations']:
			obj2 = rel['object']
			if (node, obj2) not in relations:
				relations[(node, obj2)] = set()
			relations[(node, obj2)].add(rel['name'])

	subgraph_to_relation = {}
	for (obj1, obj2), rels in relations.items():

		subgraphs1 = _enumerate_subgraphs_w_object(scene_graph, obj1, subgraph_size, exclude_object=[obj2])
		subgraphs1 = [subgraph for subgraph in subgraphs1 if subgraph_to_nodes_cnt[subgraph] == 1]

		subgraphs2 = _enumerate_subgraphs_w_object(scene_graph, obj2, subgraph_size, exclude_object=[obj1])
		subgraphs2 = [subgraph for subgraph in subgraphs2 if subgraph_to_nodes_cnt[subgraph] == 1]

		obj1_name = scene_graph["objects"][obj1]["name"]
		obj2_name = scene_graph["objects"][obj2]["name"]
		for subgraph1 in subgraphs1:
			for subgraph2 in subgraphs2:
				subgraph_to_relation[(subgraph1, subgraph2)] = (rels, obj1_name, obj2_name)

	return subgraph_to_relation


def generate_relation_qa(scene_graph, subgraph_size=3):
	subgraphs_to_rels = enumerate_relation_subgraphs(scene_graph, subgraph_size=subgraph_size)
	qas = []
	for subgraph, (rels, obj1, obj2) in subgraphs_to_rels.items():

		source_obj_reference, target_obj_reference = subgraph
		source_subgraph = json.loads(source_obj_reference)
		target_subgraph = json.loads(target_obj_reference)
		source_obj_reference = compose_object_reference(source_obj_reference)
		target_obj_reference = compose_object_reference(target_obj_reference)

		source_attributed_name = compose_attributed_name(source_subgraph.get("attributes", []), "object")
		target_attributed_name = compose_attributed_name(target_subgraph.get("attributes", []), "object")

		if source_obj_reference != "":
			source_obj_reference = f", which {source_obj_reference}"
		if target_obj_reference != "":
			target_obj_reference = f", which {target_obj_reference}"

		question = f"What is the relation from the {source_attributed_name}{source_obj_reference}, to the {target_attributed_name}{target_obj_reference}?"

		qas.append((question, list(rels)))

	return qas
