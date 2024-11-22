# conditional_clause
CONDITIONAL_PICK_FROM_MULTIPLE = ["in the list of", "among", "in", "from", "out of"]

# appear condition
CONDITIONAL_APPEAR_VERB = ["appears", "occurs", "is seen"]
CONDITIONAL_MOST_FREQUENT_ADJ = ["the most frequent", "the most commonly found", "the most frequently occurring"]
CONDITIONAL_MOST_FREQUENT_ADV = ["the most frequently", "the most commonly"]
CONDITIONAL_MOST_FREQUENT_ADJ_MULTI = ["the most instances of", "the most", "the most of", "the most number of"]
CONDITIONAL_LEAST_FREQUENT_ADJ = ["the least frequent", "the least commonly found", "the least frequently occurring"]
CONDITIONAL_LEAST_FREQUENT_ADV = ["the least frequently", "the least commonly"]
CONDITIONAL_LEAST_FREQUENT_ADJ_MULTI = ["the least instances of", "the least", "the least of", "the least number of"]

# direction condition
CONDITIONAL_DIRECTION_VERB = ["is positioned on", "is located on"]
CONDITIONAL_LEFT_ADJ = ["the most leftward", "the extreme left", "the far left"]
CONDITIONAL_LEFT_ADV = CONDITIONAL_LEFT_ADJ
CONDITIONAL_RIGHT_ADJ = ["the most rightward", "the extreme right", "the far right"]
CONDITIONAL_RIGHT_ADV = CONDITIONAL_RIGHT_ADJ
CONDITIONAL_TOP_ADJ = ["the most top side", "the extreme top", "the most upward"]
CONDITIONAL_TOP_ADV = CONDITIONAL_TOP_ADJ
CONDITIONAL_BOTTOM_ADJ = ["the most bottom side", "the extreme bottom", "the most downward"]
CONDITIONAL_BOTTOM_ADV = CONDITIONAL_BOTTOM_ADJ
# distance condition
CONDITIONAL_DISTANCE_VERB = ["is positioned", "is located"]
CONDITIONAL_CLOSER_ADJ = ["closer", "nearer", "closest"]
CONDITIONAL_CLOSER_ADV = CONDITIONAL_CLOSER_ADJ
CONDITIONAL_FARTHER_ADJ = ["farther", "farthest"]
CONDITIONAL_FARTHER_ADV = CONDITIONAL_FARTHER_ADJ + ["farther away"]
CONDITIONAL_COMPARE_VIEW_POINT = ["in depth"]
CONDITIONAL_ABSOLUTE_VIEW_POINT = CONDITIONAL_COMPARE_VIEW_POINT + ["to the camera", "to the viewpoint"]

# all images clause
CONDITIONAL_ALL_IMAGES = ["in these images", "across these images", "among these images"]

# same / different reference
CONDITIONAL_DIFFERENT_REFERENCE = ["the different objects", "the part of different objects"]
CONDITIONAL_SAME_REFERENCE = ["the same object", "the part of the same object"]

# imperative sentence
IMPERATIVE_FOLLOWED_SENTENCE = ["can you tell", "tell me", "identify", "determine"]
IMPERATIVE_FOLLOWED_PHRASE = IMPERATIVE_FOLLOWED_SENTENCE + ["provide"]

# existential clause
EXISTENTIAL_SAW = ["exist", "are available", "you see", "are there", "are visible", "present"]
EXISTENTIAL_LOCATE = ["located at", "positioned at", "found at", "present at"]



def adding_imperative_to_prompt(question_sentences, question_phrases):
	templates = []
	for question_sentence in question_sentences:
		# question_sentence can use both with and without imperative sentence

		# don't use imperative sentence
		prompt = question_sentence['prompt']
		full_prompt = (prompt + "?").capitalize()
		response = question_sentence['response']
		templates.append({"prompt": full_prompt, "response": response})

		# use imperative sentence
		for imperative in IMPERATIVE_FOLLOWED_SENTENCE:
			full_prompt = (imperative + " " + prompt + "?").capitalize()
			templates.append({"prompt": full_prompt, "response": response})

	for question_phrase in question_phrases:
		prompt = question_phrase['prompt']
		response = question_phrase['response']
		for imperative in IMPERATIVE_FOLLOWED_PHRASE:
			if imperative == "can you tell":
				full_prompt = (imperative + " " + prompt + "?").capitalize()
			else:
				full_prompt = (imperative + " " + prompt + ".").capitalize()
			templates.append({"prompt": full_prompt, "response": response})

	return templates


def get_qa_template(name):
	return eval(f"{name}_template()")


def howmany_template():
	question_sentences_structures = [
		"how many {name}",
		"how many {name} {existential_clause}",
		"what is the number of {name}",
		"what is the number of {name} that {existential_clause}",
		"what is the quantity of {name}",
		"what is the quantity of {name} that {existential_clause}",

	]
	question_phrases_structures = [
		"the number of {name}",
		"the quantity of {name}",
		"the number of {name} that {existential_clause}",
		"the quantity of {name} that {existential_clause}",
	]

	question_sentences = []
	for structure in question_sentences_structures:
		if "{existential_clause}" in structure:
			for existential_clause in EXISTENTIAL_SAW:
				prompt = structure.format(existential_clause=existential_clause, name="{name}")
				response = "{count}"
				question_sentences.append({"prompt": prompt, "response": response})
		else:
			prompt = structure
			response = "{count}"
			question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		if "{existential_clause}" in structure:
			for existential_clause in EXISTENTIAL_SAW:
				prompt = structure.format(existential_clause=existential_clause, name="{name}")
				response = "{count}"
				question_phrases.append({"prompt": prompt, "response": response})
		else:
			prompt = structure
			response = "{count}"
			question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


def ExistsObjectGenerator_template():
	return howmany_template()


def MostObjectGenerator_template():
	question_sentences_structures1 = [
		"{conditional_clause} {candidates}, which object is {frequency_adj}",
		"{conditional_clause} {candidates}, what is {frequency_adj} object",
		"which object is {frequency_adj}, {conditional_clause} {candidates}",
		"what is the {frequency_adj} object {conditional_clause} {candidates}",
	]
	question_sentences_structures2 = [
		"{conditional_clause} {candidates}, which object {appear_verb} {frequency_adv}",
		"which object {appear_verb} {frequency_adv}, {conditional_clause} {candidates}",
	]

	question_phrases_structures = [
		"{frequency_adj} object {conditional_clause} {candidates}",
	]

	question_sentences = []
	for structure in question_sentences_structures1:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for frequency_adj in CONDITIONAL_MOST_FREQUENT_ADJ:
				prompt = structure.format(conditional_clause=conditional_clause, frequency_adj=frequency_adj, candidates="{candidates}")
				response = "{name}"
				question_sentences.append({"prompt": prompt, "response": response})

	for structure in question_sentences_structures2:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for appear_verb in CONDITIONAL_APPEAR_VERB:
				for frequency_adv in CONDITIONAL_MOST_FREQUENT_ADV:
					prompt = structure.format(conditional_clause=conditional_clause, appear_verb=appear_verb, frequency_adv=frequency_adv, candidates="{candidates}")
					response = "{name}"
					question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for frequency_adj in CONDITIONAL_MOST_FREQUENT_ADJ:
				prompt = structure.format(conditional_clause=conditional_clause, frequency_adj=frequency_adj, candidates="{candidates}")
				response = "{name}"
				question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


def LeastObjectGenerator_template():
	question_sentences_structures1 = [
		"{conditional_clause} {candidates}, which object is {frequency_adj}",
		"{conditional_clause} {candidates}, what is {frequency_adj} object",
		"which object is {frequency_adj}, {conditional_clause} {candidates}",
		"what is the {frequency_adj} object {conditional_clause} {candidates}",
	]
	question_sentences_structures2 = [
		"{conditional_clause} {candidates}, which object {appear_verb} {frequency_adv}",
		"which object {appear_verb} {frequency_adv}, {conditional_clause} {candidates}",
	]

	question_phrases_structures = [
		"{frequency_adj} object {conditional_clause} {candidates}",
	]

	question_sentences = []
	for structure in question_sentences_structures1:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for frequency_adj in CONDITIONAL_LEAST_FREQUENT_ADJ:
				prompt = structure.format(conditional_clause=conditional_clause, frequency_adj=frequency_adj, candidates="{candidates}")
				response = "{name}"
				question_sentences.append({"prompt": prompt, "response": response})

	for structure in question_sentences_structures2:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for appear_verb in CONDITIONAL_APPEAR_VERB:
				for frequency_adv in CONDITIONAL_LEAST_FREQUENT_ADV:
					prompt = structure.format(conditional_clause=conditional_clause, appear_verb=appear_verb, frequency_adv=frequency_adv, candidates="{candidates}")
					response = "{name}"
					question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for frequency_adj in CONDITIONAL_LEAST_FREQUENT_ADJ:
				prompt = structure.format(conditional_clause=conditional_clause, frequency_adj=frequency_adj, candidates="{candidates}")
				response = "{name}"
				question_phrases.append({"prompt": prompt, "response": response})
	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


def LeftMostObjectGenerator_template():
	question_sentences_structures1 = [
		"{conditional_clause} {candidates}, which one is on {position_adj}",
		"{conditional_clause} {candidates}, what is {position_adj} object",
		"which object is {position_adj}, {conditional_clause} {candidates}",
		"what is the {position_adj} object {conditional_clause} {candidates}",
	]
	question_sentences_structures2 = [
		"{conditional_clause} {candidates}, which object {direction_verb} {position_adv}",
		"which object {direction_verb} {position_adv}, {conditional_clause} {candidates}",
	]
	question_phrases_structures = [
		"{position_adj} object {conditional_clause} {candidates}",
	]

	question_sentences = []
	for structure in question_sentences_structures1:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for position_adj in CONDITIONAL_LEFT_ADJ:
				prompt = structure.format(conditional_clause=conditional_clause, position_adj=position_adj, candidates="{candidates}")
				response = "{name}"
				question_sentences.append({"prompt": prompt, "response": response})

	for structure in question_sentences_structures2:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for direction_verb in CONDITIONAL_DIRECTION_VERB:
				for position_adv in CONDITIONAL_LEFT_ADV:
					prompt = structure.format(conditional_clause=conditional_clause, direction_verb=direction_verb, position_adv=position_adv, candidates="{candidates}")
					response = "{name}"
					question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for position_adj in CONDITIONAL_LEFT_ADJ:
				prompt = structure.format(conditional_clause=conditional_clause, position_adj=position_adj, candidates="{candidates}")
				response = "{name}"
				question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


def RightMostObjectGenerator_template():
	question_sentences_structures1 = [
		"{conditional_clause} {candidates}, which one is on {position_adj}",
		"{conditional_clause} {candidates}, what is {position_adj} object",
		"which object is {position_adj}, {conditional_clause} {candidates}",
		"what is the {position_adj} object {conditional_clause} {candidates}",
	]
	question_sentences_structures2 = [
		"{conditional_clause} {candidates}, which object {direction_verb} {position_adv}",
		"which object {direction_verb} {position_adv}, {conditional_clause} {candidates}",
	]
	question_phrases_structures = [
		"{position_adj} object {conditional_clause} {candidates}",
	]

	question_sentences = []
	for structure in question_sentences_structures1:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for position_adj in CONDITIONAL_RIGHT_ADJ:
				prompt = structure.format(conditional_clause=conditional_clause, position_adj=position_adj, candidates="{candidates}")
				response = "{name}"
				question_sentences.append({"prompt": prompt, "response": response})

	for structure in question_sentences_structures2:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for direction_verb in CONDITIONAL_DIRECTION_VERB:
				for position_adv in CONDITIONAL_RIGHT_ADV:
					prompt = structure.format(conditional_clause=conditional_clause, direction_verb=direction_verb, position_adv=position_adv, candidates="{candidates}")
					response = "{name}"
					question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for position_adj in CONDITIONAL_RIGHT_ADJ:
				prompt = structure.format(conditional_clause=conditional_clause, position_adj=position_adj, candidates="{candidates}")
				response = "{name}"
				question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


def TopMostObjectGenerator_template():
	question_sentences_structures1 = [
		"{conditional_clause} {candidates}, which one is on {position_adj}",
		"{conditional_clause} {candidates}, what is {position_adj} object",
		"which object is {position_adj}, {conditional_clause} {candidates}",
		"what is the {position_adj} object {conditional_clause} {candidates}",
	]
	question_sentences_structures2 = [
		"{conditional_clause} {candidates}, which object {direction_verb} {position_adv}",
		"which object {direction_verb} {position_adv}, {conditional_clause} {candidates}",
	]
	question_phrases_structures = [
		"{position_adj} object {conditional_clause} {candidates}",
	]

	question_sentences = []
	for structure in question_sentences_structures1:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for position_adj in CONDITIONAL_TOP_ADJ:
				prompt = structure.format(conditional_clause=conditional_clause, position_adj=position_adj, candidates="{candidates}")
				response = "{name}"
				question_sentences.append({"prompt": prompt, "response": response})

	for structure in question_sentences_structures2:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for direction_verb in CONDITIONAL_DIRECTION_VERB:
				for position_adv in CONDITIONAL_TOP_ADV:
					prompt = structure.format(conditional_clause=conditional_clause, direction_verb=direction_verb, position_adv=position_adv, candidates="{candidates}")
					response = "{name}"
					question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for position_adj in CONDITIONAL_TOP_ADJ:
				prompt = structure.format(conditional_clause=conditional_clause, position_adj=position_adj, candidates="{candidates}")
				response = "{name}"
				question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


def BottomMostObjectGenerator_template():
	question_sentences_structures1 = [
		"{conditional_clause} {candidates}, which one is on {position_adj}",
		"{conditional_clause} {candidates}, what is {position_adj} object",
		"which object is {position_adj}, {conditional_clause} {candidates}",
		"what is the {position_adj} object {conditional_clause} {candidates}",
	]
	question_sentences_structures2 = [
		"{conditional_clause} {candidates}, which object {direction_verb} {position_adv}",
		"which object {direction_verb} {position_adv}, {conditional_clause} {candidates}",
	]
	question_phrases_structures = [
		"{position_adj} object {conditional_clause} {candidates}",
	]

	question_sentences = []
	for structure in question_sentences_structures1:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for position_adj in CONDITIONAL_BOTTOM_ADJ:
				prompt = structure.format(conditional_clause=conditional_clause, position_adj=position_adj, candidates="{candidates}")
				response = "{name}"
				question_sentences.append({"prompt": prompt, "response": response})

	for structure in question_sentences_structures2:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for direction_verb in CONDITIONAL_DIRECTION_VERB:
				for position_adv in CONDITIONAL_BOTTOM_ADV:
					prompt = structure.format(conditional_clause=conditional_clause, direction_verb=direction_verb, position_adv=position_adv, candidates="{candidates}")
					response = "{name}"
					question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for position_adj in CONDITIONAL_BOTTOM_ADJ:
				prompt = structure.format(conditional_clause=conditional_clause, position_adj=position_adj, candidates="{candidates}")
				response = "{name}"
				question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


## attribute
def ExistsAttributeGenerator_template():
	return howmany_template()


def AttributeBBoxGenerator_template():
	question_sentences_structures = [
		"what are the attributes of the {name} {existential_clause} region {bbox}",
		"what are the attributes for the {name} {existential_clause} region {bbox}"

	]
	question_phrases_structures = [
		"the attributes of the {name} {existential_clause} region {bbox}",
		"the attributes for the {name} {existential_clause} region {bbox}",
	]

	question_sentences = []
	for structure in question_sentences_structures:
		for existential_clause in EXISTENTIAL_LOCATE:
			prompt = structure.format(existential_clause=existential_clause, name="{name}", bbox="{bbox}")
			response = "{attribute_values}"
			question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for existential_clause in EXISTENTIAL_LOCATE:
			prompt = structure.format(existential_clause=existential_clause, name="{name}", bbox="{bbox}")
			response = "{attribute_values}"
			question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


def TypedAttributeBBoxGenerator_template():
	question_sentences_structures = [
		"what is the {attribute_type} of the {name} {existential_clause} region {bbox}",
		"what is the {attribute_type} for the {name} {existential_clause} region {bbox}"

	]
	question_phrases_structures = [
		"the {attribute_type} of the {name} {existential_clause} region {bbox}",
		"the {attribute_type} for the {name} {existential_clause} region {bbox}",
	]

	question_sentences = []
	for structure in question_sentences_structures:
		for existential_clause in EXISTENTIAL_LOCATE:
			prompt = structure.format(existential_clause=existential_clause, attribute_type="{attribute_type}", name="{name}", bbox="{bbox}")
			response = "{attribute_values}"
			question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for existential_clause in EXISTENTIAL_LOCATE:
			prompt = structure.format(existential_clause=existential_clause, attribute_type="{attribute_type}", name="{name}", bbox="{bbox}")
			response = "{attribute_values}"
			question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


## relation
def ExistsRelationGenerator_template():
	question_sentences_structures = [
		"how are {object1} and {object2} related",
		"what relation exists between {object1} and {object2}",
		"what kind of relationship exists between {object1} and {object2}",
		"what is the specific relationship between {object1} and {object2}",
		"what is the relation between {object1} and {object2}"
	]
	question_phrases_structures = [
		"the relation between {object1} and {object2}",
	]

	question_sentences = []
	for structure in question_sentences_structures:
		prompt = structure.format(object1="{object1}", object2="{object2}")
		response = "{relation}"
		question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		prompt = structure.format(object1="{object1}", object2="{object2}")
		response = "{relation}"
		question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


def RelationBBoxGenerator_template():
	question_sentences_structures = [
		"how are objects at {bbox1} and {bbox2} related",
		"what relation exists between objects at {bbox1} and {bbox2}",
		"what kind of relationship exists between objects at {bbox1} and {bbox2}",
		"what is the specific relationship between objects at {bbox1} and {bbox2}",
		"what is the relation between objects at {bbox1} and {bbox2}"
	]
	question_phrases_structures = [
		"the relation between {bbox1} and {bbox2}",
	]

	question_sentences = []
	for structure in question_sentences_structures:
		prompt = structure.format(bbox1="{bbox1}", bbox2="{bbox2}")
		response = "{relation}"
		question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		prompt = structure.format(bbox1="{bbox1}", bbox2="{bbox2}")
		response = "{relation}"
		question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


def HeadRelationGenerator_template():
	question_sentences_structures = [
		"{conditional_clause} {candidates}, which one is {relation} {object2}",
		"which one is {relation} {object2} {conditional_clause} {candidates}",
		"which of {candidates} is {relation} {object2}",
	]

	question_phrases_structures = [
		"the one that {relation} {object2} {conditional_clause} {candidates}",
	]

	question_sentences = []
	for structure in question_sentences_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			prompt = structure.format(relation="{relation}", object2="{object2}", conditional_clause=conditional_clause, candidates="{candidates}")
			response = "{object1}"
			question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			prompt = structure.format(relation="{relation}", object2="{object2}", conditional_clause=conditional_clause, candidates="{candidates}")
			response = "{object1}"
			question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


## object depths
def CloserObjectGenerator_template():
	question_sentences_structures1 = [
		"{conditional_clause} {candidates}, which object is {distance_adj} {view_point}",
		"{conditional_clause} {candidates}, what is {distance_adj} object {view_point}",
		"which object is {distance_adj} {view_point}, {conditional_clause} {candidates}",
		"what is the {distance_adj} object {view_point}, {conditional_clause} {candidates}",
	]
	question_sentences_structures2 = [
		"{conditional_clause} {candidates}, which object {distance_verb} {distance_adv} {view_point}",
		"which object {distance_verb} {distance_adv} {view_point}, {conditional_clause} {candidates}",
	]
	question_phrases_structures = [
		"the {distance_adj} object {view_point}, {conditional_clause} {candidates}",
	]

	question_sentences = []
	for structure in question_sentences_structures1:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_adj in CONDITIONAL_CLOSER_ADJ:
				for view_point in CONDITIONAL_ABSOLUTE_VIEW_POINT:
					prompt = structure.format(conditional_clause=conditional_clause, distance_adj=distance_adj, view_point=view_point,
											  candidates="{candidates}")
					response = "{closer}"
					question_sentences.append({"prompt": prompt, "response": response})

	for structure in question_sentences_structures2:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_verb in CONDITIONAL_DISTANCE_VERB:
				for distance_adv in CONDITIONAL_CLOSER_ADV:
					for view_point in CONDITIONAL_ABSOLUTE_VIEW_POINT:
						prompt = structure.format(conditional_clause=conditional_clause, distance_verb=distance_verb, distance_adv=distance_adv,
												  view_point=view_point, candidates="{candidates}")
						response = "{closer}"
						question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_adj in CONDITIONAL_CLOSER_ADJ:
				for view_point in CONDITIONAL_ABSOLUTE_VIEW_POINT:
					prompt = structure.format(conditional_clause=conditional_clause, distance_adj=distance_adj,
											  view_point=view_point, candidates="{candidates}")
					response = "{closer}"
					question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


def FartherObjectGenerator_template():
	question_sentences_structures1 = [
		"{conditional_clause} {candidates}, which object is {distance_adj} {view_point}",
		"{conditional_clause} {candidates}, what is {distance_adj} object {view_point}",
		"which object is {distance_adj} {view_point}, {conditional_clause} {candidates}",
		"what is the {distance_adj} object {view_point}, {conditional_clause} {candidates}",
	]
	question_sentences_structures2 = [
		"{conditional_clause} {candidates}, which object {distance_verb} {distance_adv} {view_point}",
		"which object {distance_verb} {distance_adv} {view_point}, {conditional_clause} {candidates}",
	]
	question_phrases_structures = [
		"the {distance_adj} object {view_point}, {conditional_clause} {candidates}",
	]

	question_sentences = []
	for structure in question_sentences_structures1:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_adj in CONDITIONAL_FARTHER_ADJ:
				for view_point in CONDITIONAL_ABSOLUTE_VIEW_POINT:
					prompt = structure.format(conditional_clause=conditional_clause, distance_adj=distance_adj, view_point=view_point,
											  candidates="{candidates}")
					response = "{farther}"
					question_sentences.append({"prompt": prompt, "response": response})

	for structure in question_sentences_structures2:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_verb in CONDITIONAL_DISTANCE_VERB:
				for distance_adv in CONDITIONAL_FARTHER_ADV:
					for view_point in CONDITIONAL_ABSOLUTE_VIEW_POINT:
						prompt = structure.format(conditional_clause=conditional_clause, distance_verb=distance_verb, distance_adv=distance_adv,
												  view_point=view_point, candidates="{candidates}")
						response = "{farther}"
						question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_adj in CONDITIONAL_FARTHER_ADJ:
				for view_point in CONDITIONAL_ABSOLUTE_VIEW_POINT:
					prompt = structure.format(conditional_clause=conditional_clause, distance_adj=distance_adj,
											  view_point=view_point, candidates="{candidates}")
					response = "{farther}"
					question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


def CloserToAnchorObjectGenerator_template():
	question_sentences_structures1 = [
		"{conditional_clause} {candidates}, which object is {distance_adj} to the {anchor} {view_point}",
		"{conditional_clause} {candidates}, what is {distance_adj} object to the {anchor} {view_point}",
		"which object is {distance_adj} to the {anchor} {view_point}, {conditional_clause} {candidates}",
		"what is the {distance_adj} object to the {anchor} {view_point}, {conditional_clause} {candidates}",
	]

	question_sentences_structures2 = [
		"{conditional_clause} {candidates}, which object {distance_verb} {distance_adv} to the {anchor} {view_point}",
		"which object {distance_verb} {distance_adv} to the {anchor} {view_point}, {conditional_clause} {candidates}",
	]
	question_phrases_structures = [
		"the {distance_adj} object to the {anchor} {view_point}, {conditional_clause} {candidates}",
	]

	question_sentences = []
	for structure in question_sentences_structures1:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_adj in CONDITIONAL_CLOSER_ADJ:
				for view_point in CONDITIONAL_COMPARE_VIEW_POINT:
					prompt = structure.format(conditional_clause=conditional_clause, distance_adj=distance_adj,
											  view_point=view_point, anchor="{anchor}", candidates="{candidates}")
					response = "{closer}"
					question_sentences.append({"prompt": prompt, "response": response})

	for structure in question_sentences_structures2:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_verb in CONDITIONAL_DISTANCE_VERB:
				for distance_adv in CONDITIONAL_CLOSER_ADV:
					for view_point in CONDITIONAL_COMPARE_VIEW_POINT:
						prompt = structure.format(conditional_clause=conditional_clause, distance_verb=distance_verb, distance_adv=distance_adv,
												  view_point=view_point, anchor="{anchor}", candidates="{candidates}")
						response = "{closer}"
						question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_adj in CONDITIONAL_CLOSER_ADJ:
				for view_point in CONDITIONAL_COMPARE_VIEW_POINT:
					prompt = structure.format(conditional_clause=conditional_clause, distance_adj=distance_adj,
											  view_point=view_point, anchor="{anchor}", candidates="{candidates}")
					response = "{closer}"
					question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


def FartherToAnchorObjectGenerator_template():
	question_sentences_structures1 = [
		"{conditional_clause} {candidates}, which object is {distance_adj} to the {anchor} {view_point}",
		"{conditional_clause} {candidates}, what is {distance_adj} object to the {anchor} {view_point}",
		"which object is {distance_adj} to the {anchor} {view_point}, {conditional_clause} {candidates}",
		"what is the {distance_adj} object to the {anchor} {view_point}, {conditional_clause} {candidates}",
	]

	question_sentences_structures2 = [
		"{conditional_clause} {candidates}, which object {distance_verb} {distance_adv} to the {anchor} {view_point}",
		"which object {distance_verb} {distance_adv} to the {anchor} {view_point}, {conditional_clause} {candidates}",
	]
	question_phrases_structures = [
		"the {distance_adj} object to the {anchor} {view_point}, {conditional_clause} {candidates}",
	]

	question_sentences = []
	for structure in question_sentences_structures1:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_adj in CONDITIONAL_FARTHER_ADJ:
				for view_point in CONDITIONAL_COMPARE_VIEW_POINT:
					prompt = structure.format(conditional_clause=conditional_clause, distance_adj=distance_adj,
											  view_point=view_point, anchor="{anchor}", candidates="{candidates}")
					response = "{farther}"
					question_sentences.append({"prompt": prompt, "response": response})

	for structure in question_sentences_structures2:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_verb in CONDITIONAL_DISTANCE_VERB:
				for distance_adv in CONDITIONAL_FARTHER_ADV:
					for view_point in CONDITIONAL_COMPARE_VIEW_POINT:
						prompt = structure.format(conditional_clause=conditional_clause, distance_verb=distance_verb, distance_adv=distance_adv,
												  view_point=view_point, anchor="{anchor}", candidates="{candidates}")
						response = "{farther}"
						question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_adj in CONDITIONAL_FARTHER_ADJ:
				for view_point in CONDITIONAL_COMPARE_VIEW_POINT:
					prompt = structure.format(conditional_clause=conditional_clause, distance_adj=distance_adj,
											  view_point=view_point, anchor="{anchor}", candidates="{candidates}")
					response = "{farther}"
					question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


## Depth
def CloserPointGenerator_template():
	question_sentences_structures1 = [
		"{conditional_clause} {candidates}, which point is {distance_adj} {view_point}",
		"{conditional_clause} {candidates}, what is {distance_adj} point {view_point}",
		"which point is {distance_adj} {view_point}, {conditional_clause} {candidates}",
		"what is the {distance_adj} point {view_point}, {conditional_clause} {candidates}",
	]
	question_sentences_structures2 = [
		"{conditional_clause} {candidates}, which point {distance_verb} {distance_adv} {view_point}",
		"which point {distance_verb} {distance_adv} {view_point}, {conditional_clause} {candidates}",
	]
	question_phrases_structures = [
		"the {distance_adj} point {view_point}, {conditional_clause} {candidates}",
	]

	question_sentences = []
	for structure in question_sentences_structures1:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_adj in CONDITIONAL_CLOSER_ADJ:
				for view_point in CONDITIONAL_ABSOLUTE_VIEW_POINT:
					prompt = structure.format(conditional_clause=conditional_clause, distance_adj=distance_adj, view_point=view_point,
											  candidates="{candidates}")
					response = "{closer}"
					question_sentences.append({"prompt": prompt, "response": response})

	for structure in question_sentences_structures2:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_verb in CONDITIONAL_DISTANCE_VERB:
				for distance_adv in CONDITIONAL_CLOSER_ADV:
					for view_point in CONDITIONAL_ABSOLUTE_VIEW_POINT:
						prompt = structure.format(conditional_clause=conditional_clause, distance_verb=distance_verb, distance_adv=distance_adv,
												  view_point=view_point, candidates="{candidates}")
						response = "{closer}"
						question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_adj in CONDITIONAL_CLOSER_ADJ:
				for view_point in CONDITIONAL_ABSOLUTE_VIEW_POINT:
					prompt = structure.format(conditional_clause=conditional_clause, distance_adj=distance_adj,
											  view_point=view_point, candidates="{candidates}")
					response = "{closer}"
					question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


def FartherPointGenerator_template():
	question_sentences_structures1 = [
		"{conditional_clause} {candidates}, which point is {distance_adj} {view_point}",
		"{conditional_clause} {candidates}, what is {distance_adj} point {view_point}",
		"which point is {distance_adj} {view_point}, {conditional_clause} {candidates}",
		"what is the {distance_adj} point {view_point}, {conditional_clause} {candidates}",
	]
	question_sentences_structures2 = [
		"{conditional_clause} {candidates}, which point {distance_verb} {distance_adv} {view_point}",
		"which point {distance_verb} {distance_adv} {view_point}, {conditional_clause} {candidates}",
	]
	question_phrases_structures = [
		"the {distance_adj} point {view_point}, {conditional_clause} {candidates}",
	]

	question_sentences = []
	for structure in question_sentences_structures1:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_adj in CONDITIONAL_FARTHER_ADJ:
				for view_point in CONDITIONAL_ABSOLUTE_VIEW_POINT:
					prompt = structure.format(conditional_clause=conditional_clause, distance_adj=distance_adj, view_point=view_point,
											  candidates="{candidates}")
					response = "{farther}"
					question_sentences.append({"prompt": prompt, "response": response})

	for structure in question_sentences_structures2:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_verb in CONDITIONAL_DISTANCE_VERB:
				for distance_adv in CONDITIONAL_FARTHER_ADV:
					for view_point in CONDITIONAL_ABSOLUTE_VIEW_POINT:
						prompt = structure.format(conditional_clause=conditional_clause, distance_verb=distance_verb, distance_adv=distance_adv,
												  view_point=view_point, candidates="{candidates}")
						response = "{farther}"
						question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for distance_adj in CONDITIONAL_FARTHER_ADJ:
				for view_point in CONDITIONAL_ABSOLUTE_VIEW_POINT:
					prompt = structure.format(conditional_clause=conditional_clause, distance_adj=distance_adj,
											  view_point=view_point, candidates="{candidates}")
					response = "{farther}"
					question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


## Segment
def SameObjectSegGenerator_template():
	question_sentences_structures = [
		"{conditional_clause} {candidates}, which point is in {reference} as {anchor_point}",
		"which point from {candidates} is in {reference} as {anchor_point}",
		"which of {candidates} is in {reference} as {anchor_point}",
		"which point {conditional_clause} {candidates} belong to {reference} as {anchor_point}",
	]

	question_phrases_structures = [
		"the point that is in {reference} as {anchor_point} {conditional_clause} {candidates}",
	]

	question_sentences = []
	for structure in question_sentences_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for reference in CONDITIONAL_SAME_REFERENCE:
				prompt = structure.format(conditional_clause=conditional_clause, reference=reference, candidates="{candidates}", anchor_point="{anchor_point}")
				response = "{same_point}"
				question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for reference in CONDITIONAL_SAME_REFERENCE:
				prompt = structure.format(conditional_clause=conditional_clause, reference=reference, candidates="{candidates}", anchor_point="{anchor_point}")
				response = "{same_point}"
				question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


def DiffObjectSegGenerator_template():
	question_sentences_structures = [
		"{conditional_clause} {candidates}, which point is in {reference} as {anchor_point}",
		"which point from {candidates} is in {reference} as {anchor_point}",
		"which of {candidates} is in {reference} as {anchor_point}",
		"which point {conditional_clause} {candidates} belong to {reference} as {anchor_point}",
	]

	question_phrases_structures = [
		"the point that is in {reference} as {anchor_point} {conditional_clause} {candidates}",
	]

	question_sentences = []
	for structure in question_sentences_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for reference in CONDITIONAL_DIFFERENT_REFERENCE:
				prompt = structure.format(conditional_clause=conditional_clause, reference=reference, candidates="{candidates}", anchor_point="{anchor_point}")
				response = "{diff_point}"
				question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for conditional_clause in CONDITIONAL_PICK_FROM_MULTIPLE:
			for reference in CONDITIONAL_DIFFERENT_REFERENCE:
				prompt = structure.format(conditional_clause=conditional_clause, reference=reference, candidates="{candidates}", anchor_point="{anchor_point}")
				response = "{diff_point}"
				question_phrases.append({"prompt": prompt, "response": response})
	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


# Multi Image
def HasObjectMultiGenerator_template():
	question_sentences_structures = [
		"which image has {object}",
		"which image contains {object}",
		"which image shows {object}"
	]

	question_sentences = []
	for structure in question_sentences_structures:
		prompt = structure.format(object = "{object}")
		response = "{answer}"
		question_sentences.append({"prompt": prompt, "response": response})

	# this generator doesn't have question_phrases type of questions
	question_phrases = []

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


def HasAttributedObjectMultiGenerator_template():
	question_sentences_structures = [
		"which image has {attribute} {object}",
		"which image contains {attribute} {object}",
		"which image shows {attribute} {object}"
	]

	question_sentences = []
	for structure in question_sentences_structures:
		prompt = structure.format(object = "{object}", attribute = "{attribute}")
		response = "{answer}"
		question_sentences.append({"prompt": prompt, "response": response})

	# this generator doesn't have question_phrases type of questions
	question_phrases = []

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates

def HasNotObjectMultiGenerator_template():
	question_sentences_structures = [
		"which image doesn't have {object}",
		"which image doesn't contain {object}",
		"which image doesn't show {object}"
	]

	question_sentences = []
	for structure in question_sentences_structures:
		prompt = structure.format(object = "{object}")
		response = "{answer}"
		question_sentences.append({"prompt": prompt, "response": response})

	# this generator doesn't have question_phrases type of questions
	question_phrases = []
	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates

def HasNotAttributedObjectMultiGenerator_template():
	question_sentences_structures = [
		"which image doesn't have {attribute} {object}",
		"which image doesn't contain {attribute} {object}",
		"which image doesn't show {attribute} {object}"
	]

	question_sentences = []
	for structure in question_sentences_structures:
		prompt = structure.format(object = "{object}", attribute = "{attribute}")
		response = "{answer}"
		question_sentences.append({"prompt": prompt, "response": response})

	# this generator doesn't have question_phrases type of questions
	question_phrases = []

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates

def HasMostObjectMultiGenerator_template():
	question_sentences_structures = [
		"which image has {frequency_adj} {object}",
		"which image contains {frequency_adj} {object}",
		"which image shows {frequency_adj} {object}"
	]

	question_sentences = []
	for structure in question_sentences_structures:
		for frequency_adj in CONDITIONAL_MOST_FREQUENT_ADJ_MULTI:
			prompt = structure.format(object = "{object}", frequency_adj = frequency_adj)
			response = "{answer}"
			question_sentences.append({"prompt": prompt, "response": response})
	# this generator doesn't have question_phrases type of questions
	question_phrases = []

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates

def HasLeastObjectMultiGenerator_template():
	question_sentences_structures = [
		"which image has {frequency_adj} {object}",
		"which image contains {frequency_adj} {object}",
		"which image shows {frequency_adj} {object}"
	]

	question_sentences = []
	for structure in question_sentences_structures:
		for frequency_adj in CONDITIONAL_LEAST_FREQUENT_ADJ_MULTI:
			prompt = structure.format(object = "{object}", frequency_adj = frequency_adj)
			response = "{answer}"
			question_sentences.append({"prompt": prompt, "response": response})
	# this generator doesn't have question_phrases type of questions
	question_phrases = []

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates

def HasRelationMultiGenerator_template():
	question_sentences_structures = [
		"in which image {object1} is {relation} {object2}",
		"which image shows that {object1} is {relation} {object2}",
	]

	question_sentences = []
	for structure in question_sentences_structures:
		prompt = structure.format(object1 = "{object1}", relation = "{relation}", object2 = "{object2}")
		response = "{answer}"
		question_sentences.append({"prompt": prompt, "response": response})
	# this generator doesn't have question_phrases type of questions
	question_phrases = []

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates

def HasNotRelationMultiGenerator_template():
	question_sentences_structures = [
		"in which image {object1} isn't {relation} {object2}",
		"which image doesn't show that {object1} is {relation} {object2}",
	]

	question_sentences = []
	for structure in question_sentences_structures:
		prompt = structure.format(object1 = "{object1}", relation = "{relation}", object2 = "{object2}")
		response = "{answer}"
		question_sentences.append({"prompt": prompt, "response": response})
	# this generator doesn't have question_phrases type of questions
	question_phrases = []

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates

# multi images aggregation
def CommonObjectMultiGenerator_template():
	question_sentences_structures = [
		"what are common objects {all_images}",
		"what are the objects that are common {all_images}",
		"which objects do these images have in common",
		"what objects appear in all of these images",
		"what objects are seen in all of these images",
	]
	question_phrases_structures = [
		"the common objects {all_images}",
		"the objects that are common {all_images}",
		"the objects that appear in all of these images",
		"the objects that are seen in all of these images",
	]


	question_sentences = []
	for structure in question_sentences_structures:
		for all_images in CONDITIONAL_ALL_IMAGES:
			prompt = structure.format(all_images = all_images)
			response = "{objects}"
			question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for all_images in CONDITIONAL_ALL_IMAGES:
			prompt = structure.format(all_images = all_images)
			response = "{objects}"
			question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates	

def CommonAttributeMultiGenerator_template():
	question_sentences_structures = [
		"what is common attribute of {object} {all_images}",
		"what is a similar attribute among these {object} {all_images}",
		"what attribute is consistent across these {object} {all_images}"
	]
	question_phrases_structures = [
		"the common attribute of {object} {all_images}",
		"the similar attribute among these {object} {all_images}",
		"the attribute that is consistent across these {object} {all_images}"
	]

	question_sentences = []
	for structure in question_sentences_structures:
		for all_images in CONDITIONAL_ALL_IMAGES:
			prompt = structure.format(object="{object}", all_images=all_images)
			response = "{attributes}"
			question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for all_images in CONDITIONAL_ALL_IMAGES:
			prompt = structure.format(object="{object}", all_images = all_images)
			response = "{attributes}"
			question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates

def CountObjectMultiGenerator_template():
	question_sentences_structures = [
		"how many {object} {all_images}",
		"how many {object} {existential_clause} {all_images}",
		"what is the number of {object} {all_images}",
		"what is the number of {object} that {existential_clause} {all_images}",
		"what is the quantity of {object} {all_images}",
		"what is the quantity of {object} that {existential_clause} {all_images}",
	]
	question_phrases_structures = [
		"the number of {object} {all_images}",
		"the quantity of {object} {all_images}",
		"the number of {object} that {existential_clause} {all_images}",
		"the quantity of {object} that {existential_clause} {all_images}",
	]

	question_sentences = []
	for structure in question_sentences_structures:
		for existential_clause in EXISTENTIAL_SAW:
			for all_images in CONDITIONAL_ALL_IMAGES:
				prompt = structure.format(existential_clause=existential_clause, object="{object}", all_images=all_images)
				response = "{answer}"
				question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for existential_clause in EXISTENTIAL_SAW:
			for all_images in CONDITIONAL_ALL_IMAGES:
				prompt = structure.format(existential_clause=existential_clause, object="{object}", all_images=all_images)
				response = "{answer}"
				question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates

def CountAttributeObjectMultiGenerator_template():
	question_sentences_structures = [
		"how many {attribute} {object} {all_images}",
		"how many {attribute} {object} {existential_clause} {all_images}",
		"what is the number of {attribute} {object} {all_images}",
		"what is the number of {attribute} {object} that {existential_clause} {all_images}",
		"what is the quantity of {attribute} {object} {all_images}",
		"what is the quantity of {attribute} {object} that {existential_clause} {all_images}",
	]
	question_phrases_structures = [
		"the number of {attribute} {object} {all_images}",
		"the quantity of {attribute} {object} {all_images}",
		"the number of {attribute} {object} that {existential_clause} {all_images}",
		"the quantity of {attribute} {object} that {existential_clause} {all_images}",
	]

	question_sentences = []
	for structure in question_sentences_structures:
		for existential_clause in EXISTENTIAL_SAW:
			for all_images in CONDITIONAL_ALL_IMAGES:
				prompt = structure.format(existential_clause=existential_clause, object="{object}", attribute="{attribute}", all_images=all_images)
				response = "{answer}"
				question_sentences.append({"prompt": prompt, "response": response})

	question_phrases = []
	for structure in question_phrases_structures:
		for existential_clause in EXISTENTIAL_SAW:
			for all_images in CONDITIONAL_ALL_IMAGES:
				prompt = structure.format(existential_clause=existential_clause, object="{object}", attribute="{attribute}", all_images=all_images)
				response = "{answer}"
				question_phrases.append({"prompt": prompt, "response": response})

	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


# multi images compare
def CompareAttributeMultiGenerator_template():
	question_sentences_structures = [
		"what is the difference of attributes of {object} {all_images}",
		"what differences can be observed in the attributes of {object} {all_images}",
		"How do the attributes of {object} compare {all_images}"

	]
	question_phrases_structures = [
		"the difference of attributes of {object} {all_images}",
	]

	question_sentences = []
	for structure in question_sentences_structures:
		for all_images in CONDITIONAL_ALL_IMAGES:
			prompt = structure.format(object="{object}", all_images=all_images)
			response = "{answer}"
			question_sentences.append({"prompt": prompt, "response": response})
	
	question_phrases = []
	for structure in question_phrases_structures:
		for all_images in CONDITIONAL_ALL_IMAGES:
			prompt = structure.format(object="{object}", all_images=all_images)
			response = "{answer}"
			question_phrases.append({"prompt": prompt, "response": response})
	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates


def CompareRelationMultiGenerator_template():
	question_sentences_structures = [
		"what is the difference of the relation between {object1} and {object2} {all_images}",
		"what differences can be observed in the relation between {object1} and {object2}",
		"How do the relation between {object1} and {object2} compare {all_images}"
	]
	question_phrases_structures = [
		"the difference of the relation between {object1} and {object2} {all_images}",
	]

	question_sentences = []
	for structure in question_sentences_structures:
		for all_images in CONDITIONAL_ALL_IMAGES:
			prompt = structure.format(object1="{object1}", object2="{object2}", all_images=all_images)
			response = "{answer}"
			question_sentences.append({"prompt": prompt, "response": response})
	
	question_phrases = []
	for structure in question_phrases_structures:
		for all_images in CONDITIONAL_ALL_IMAGES:
			prompt = structure.format(object1="{object1}", object2="{object2}", all_images=all_images)
			response = "{answer}"
			question_phrases.append({"prompt": prompt, "response": response})
	templates = adding_imperative_to_prompt(question_sentences, question_phrases)
	return templates