from collections import defaultdict

import inflect
import networkx as nx

from ..utils import make_and_description

inflect_engine = inflect.engine()


def label_repeated_objects_in_sg(graph: nx.DiGraph):
	# this function is to find the same objects in the scene_graph, like there are 2 "apple" in the sg.
	# then in the caption we can refer to them as "the first apple" and "the second apple" so it won't be confusing.
	grouped_nodes = defaultdict(list)
	for node in graph.nodes:
		value = graph.nodes[node].get("value")
		attributes = graph.nodes[node].get("attributes")
		key = (value, tuple(attributes))
		grouped_nodes[key].append(node)

	same_nodes_groups = {
		key: nodes for key, nodes in grouped_nodes.items() if len(nodes) > 1
	}
	for nodes in same_nodes_groups.values():
		for index, node in enumerate(nodes):
			graph.nodes[node]["is_repeated"] = index
	for node in graph.nodes:
		if "is_repeated" not in graph.nodes[node]:
			graph.nodes[node]["is_repeated"] = "no"

	return same_nodes_groups


def topsort(graph: nx.DiGraph):
	try:
		topo_order = list(nx.topological_sort(graph))
		return topo_order
	except nx.NetworkXUnfeasible:
		return sorted(graph.nodes, key=lambda x: graph.degree[x], reverse=True)


def mention_node(graph, node):
	if "mentioned" not in graph.nodes[node]:
		graph.nodes[node]["mentioned"] = True


def get_attr_obj_desc(graph, node, rng) -> str:
	name = graph.nodes[node]["value"]

	object_desc = ""
	if graph.nodes[node]["is_repeated"] != "no":
		object_desc += inflect_engine.ordinal(graph.nodes[node]["is_repeated"] + 1) + " "

	attrs = graph.nodes[node].get("attributes")
	attributes_desc = make_and_description(attrs, rng)
	if attributes_desc != "":
		object_desc += attributes_desc + " "

	object_desc += name

	if "mentioned" not in graph.nodes[node] and graph.nodes[node]["is_repeated"] == "no":
		if not inflect_engine.singular_noun(name):
			object_desc = inflect_engine.a(object_desc)
	else:
		object_desc = "the" + " " + object_desc

	return object_desc


def get_relation_desc(graph, node, rng) -> str:
	name = graph.nodes[node]["value"]
	if inflect_engine.singular_noun(name):
		be = "are"
	else:
		be = "is"

	# value of object and attr will be object_name[category], the .split("[")[0] means removed the category part when display them.
	relations_desc = []
	relation_to_targets = defaultdict(list)
	for head, target, data in graph.out_edges([node], data=True):
		relation_to_targets[data["value"]].append(target)

	for relation, targets in relation_to_targets.items():
		# add mentioned flag to both nodes
		mention_node(graph, node)
		for target in targets:
			mention_node(graph, target)

		targets = [get_attr_obj_desc(graph, target, rng) for target in targets]
		target_desc = make_and_description(targets, rng)

		relations_desc.append(
			f"{be} {relation} {target_desc}"
		)

	return make_and_description(relations_desc, rng)


def get_sg_desc(scene_graph, rng):
	label_repeated_objects_in_sg(scene_graph)
	topsort_order = topsort(scene_graph)
	templates = []
	for node in topsort_order:
		attr_obj = get_attr_obj_desc(scene_graph, node, rng)
		relations_desc = get_relation_desc(scene_graph, node, rng)
		if relations_desc != "":
			templates.append(attr_obj + " " + relations_desc)
		else:
			if 'mentioned' not in scene_graph.nodes[node]:
				templates.append("there is " + attr_obj)

	return ";\n".join(templates) + '.'
