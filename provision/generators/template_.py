PICK_FROM_MULTIPLE = ["in the list of", "among", "of", "in", "from", "out of"]
PICK_FROM_TWO = PICK_FROM_MULTIPLE + ['between']
MOST_FREQUENT = ["the most frequent", "most commonly found", "most frequently occurring"]
LEAST_FREQUENT = ["the least frequent", "least commonly found", "least frequently occurring"]


def get_qa_template(name):
	return eval(f"{name}_template()")


## object

def ExistsObjectGenerator_template():
	return [
		{
			"prompt"  : "How many {name}?",
			"response": "{count}"
		},
		{
			"prompt"  : "What is the number of {name}?",
			"response": "{count}"
		},
		{
			"prompt"  : "Count the number of {name}.",
			"response": "{count}"
		},
		{
			"prompt"  : "How many {name} can you see?",
			"response": "{count}"
		},
		{
			"prompt"  : "Tell me the count of {name}.",
			"response": "{count}"
		},
		{
			"prompt"  : "Please provide the number of {name}.",
			"response": "{count}"
		},
		{
			"prompt"  : "What is the total count of {name}?",
			"response": "{count}"
		},
		{
			"prompt"  : "How many instances of {name} are present?",
			"response": "{count}"
		},
		{
			"prompt"  : "What is the quantity of {name}?",
			"response": "{count}"
		},
		{
			"prompt"  : "Can you tell me how many {name} there are?",
			"response": "{count}"
		},
		{
			"prompt"  : "What's the number of {name} present?",
			"response": "{count}"
		},
		{
			"prompt"  : "How many {name} do you see?",
			"response": "{count}"
		},
		{
			"prompt"  : "What is the count of {name} available?",
			"response": "{count}"
		},
		{
			"prompt"  : "How many {name} exist?",
			"response": "{count}"
		},
		{
			"prompt"  : "How many {name} are visible?",
			"response": "{count}"
		}
	]


def MostObjectGenerator_template():
	return [
		{
			"prompt"  : "Among {candidates}, which is the most frequent object?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which object is the most frequent among {candidates}?",
			"response": "{name}"
		},
		{
			"prompt"  : "In the list of {candidates}, which object is seen the most?",
			"response": "{name}"
		},
		{
			"prompt"  : "From {candidates}, which object occurs most frequently?",
			"response": "{name}"
		},
		{
			"prompt"  : "Among {candidates}, which is the most commonly found object?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which object appears most often among {candidates}?",
			"response": "{name}"
		},
		{
			"prompt"  : "Identify the most frequent object among {candidates}.",
			"response": "{name}"
		},
		{
			"prompt"  : "Out of {candidates}, which object is most frequent?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which is the most frequently occurring object among {candidates}?",
			"response": "{name}"
		},
		{
			"prompt"  : "In {candidates}, which object is the most frequent?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which object, among {candidates}, is seen the most often?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which object is the most common among {candidates}?",
			"response": "{name}"
		}
	]


def LeastObjectGenerator_template():
	return [
		{
			"prompt"  : "Among {candidates}, which object appears the least?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which object is the least frequent among {candidates}?",
			"response": "{name}"
		},
		{
			"prompt"  : "In the list of {candidates}, which object is seen the least?",
			"response": "{name}"
		},
		{
			"prompt"  : "From {candidates}, which object occurs least frequently?",
			"response": "{name}"
		},
		{
			"prompt"  : "Among {candidates}, which is the least commonly found object?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which object appears least often among {candidates}?",
			"response": "{name}"
		},
		{
			"prompt"  : "Identify the least frequent object among {candidates}.",
			"response": "{name}"
		},
		{
			"prompt"  : "Out of {candidates}, which object is least frequent?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which is the least frequently occurring object among {candidates}?",
			"response": "{name}"
		},
		{
			"prompt"  : "In {candidates}, which object is the least frequent?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which object, among {candidates}, is seen the least often?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which object is the least common among {candidates}?",
			"response": "{name}"
		}
	]


def LeftMostObjectGenerator_template():
	return [
		{
			"prompt"  : "Among {candidates}, which is on the most left side?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which of {candidates} is positioned most to the left?",
			"response": "{name}"
		},
		{
			"prompt"  : "Out of {candidates}, which one is on the far left?",
			"response": "{name}"
		},
		{
			"prompt"  : "Among {candidates}, which is located on the leftmost side?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which object among {candidates} is the most leftward?",
			"response": "{name}"
		},
		{
			"prompt"  : "In {candidates}, which object is on the extreme left?",
			"response": "{name}"
		},
		{
			"prompt"  : "Of {candidates}, which is the most left-side object?",
			"response": "{name}"
		},
		{
			"prompt"  : "Identify the leftmost object among {candidates}.",
			"response": "{name}"
		},
		{
			"prompt"  : "Which of the following objects is on the far left: {candidates}?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which object is the leftmost among {candidates}?",
			"response": "{name}"
		}
	]


def RightMostObjectGenerator_template():
	return [
		{
			"prompt"  : "Among {candidates}, which is on the most right side?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which of {candidates} is positioned most to the right?",
			"response": "{name}"
		},
		{
			"prompt"  : "Out of {candidates}, which one is on the far right?",
			"response": "{name}"
		},
		{
			"prompt"  : "Among {candidates}, which is located on the rightmost side?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which object among {candidates} is the most rightward?",
			"response": "{name}"
		},
		{
			"prompt"  : "In {candidates}, which object is on the extreme right?",
			"response": "{name}"
		},
		{
			"prompt"  : "Of {candidates}, which is the most right-side object?",
			"response": "{name}"
		},
		{
			"prompt"  : "Identify the rightmost object among {candidates}.",
			"response": "{name}"
		},
		{
			"prompt"  : "Which of the following objects is on the far right: {candidates}?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which object is the rightmost among {candidates}?",
			"response": "{name}"
		}
	]


def TopMostObjectGenerator_template():
	return [
		{
			"prompt"  : "Among {candidates}, which is on the most top side?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which of {candidates} is positioned most to the top?",
			"response": "{name}"
		},
		{
			"prompt"  : "Out of {candidates}, which one is on the top?",
			"response": "{name}"
		},
		{
			"prompt"  : "Among {candidates}, which is located on the topmost side?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which object among {candidates} is the most upward?",
			"response": "{name}"
		},
		{
			"prompt"  : "In {candidates}, which object is on the extreme top?",
			"response": "{name}"
		},
		{
			"prompt"  : "Of {candidates}, which is the most top-side object?",
			"response": "{name}"
		},
		{
			"prompt"  : "Identify the topmost object among {candidates}.",
			"response": "{name}"
		},
		{
			"prompt"  : "Which of the following objects is on the far top: {candidates}?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which object is the topmost among {candidates}?",
			"response": "{name}"
		}
	]


def BottomMostObjectGenerator_template():
	return [
		{
			"prompt"  : "Among {candidates}, which is on the most bottom side?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which of {candidates} is positioned most to the bottom?",
			"response": "{name}"
		},
		{
			"prompt"  : "Out of {candidates}, which one is on the bottom?",
			"response": "{name}"
		},
		{
			"prompt"  : "Among {candidates}, which is located on the bottommost side?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which object among {candidates} is the most downward?",
			"response": "{name}"
		},
		{
			"prompt"  : "In {candidates}, which object is on the extreme bottom?",
			"response": "{name}"
		},
		{
			"prompt"  : "Of {candidates}, which is the most bottom-side object?",
			"response": "{name}"
		},
		{
			"prompt"  : "Identify the bottommost object among {candidates}.",
			"response": "{name}"
		},
		{
			"prompt"  : "Which of the following objects is on the far bottom: {candidates}?",
			"response": "{name}"
		},
		{
			"prompt"  : "Which object is the bottommost among {candidates}?",
			"response": "{name}"
		}
	]


## attribute

def ExistsAttributeGenerator_template():
	return [
		{
			"prompt"  : "How many {name} are there?",
			"response": "{count}"
		},
		{
			"prompt"  : "Can you tell me the number of {name}?",
			"response": "{count}"
		},
		{
			"prompt"  : "What is the count of {name}?",
			"response": "The count of {name} is {count}."
		},
		{
			"prompt"  : "Please provide the quantity of {name}.",
			"response": "{count}"
		},
		{
			"prompt"  : "How many {name} exist?",
			"response": "{count}"
		},
		{
			"prompt"  : "What is the total number of {name}?",
			"response": "{count}"
		},
		{
			"prompt"  : "How much {name} is there?",
			"response": "{count}"
		},
		{
			"prompt"  : "Count the {name} for me.",
			"response": "{count}"
		},
		{
			"prompt"  : "How many instances of {name} are there?",
			"response": "{count}"
		},
		{
			"prompt"  : "Could you give me the number of {name}?",
			"response": "{count}"
		},
		{
			"prompt"  : "Tell me how many {name} you see.",
			"response": "{count}"
		},
		{
			"prompt"  : "What is the number of {name} available?",
			"response": "{count}"
		}
	]


def AttributeBBoxGenerator_template():
	return [
		{
			"prompt"  : "What are attributes of {name} at region {bbox}?",
			"response": "{attribute_values}"
		},
		{
			"prompt"  : "Can you tell the attributes of {name} located at region {bbox}?",
			"response": "{attribute_values}"
		},
		{
			"prompt"  : "Identify the attributes of the {name} at region {bbox}.",
			"response": "{attribute_values}"
		},
		{
			"prompt"  : "List the attributes of {name} found at region {bbox}.",
			"response": "{attribute_values}"
		},
		{
			"prompt"  : "Give the attributes of {name} present at region {bbox}.",
			"response": "{attribute_values}"
		},
		{
			"prompt"  : "Tell me the attributes for the {name} positioned at region {bbox}.",
			"response": "{attribute_values}"
		}
	]


def TypedAttributeBBoxGenerator_template():
	return [
		{
			"prompt"  : "What are {attribute_type} of {name} at region {bbox}?",
			"response": "{attribute_values}"
		},
		{
			"prompt"  : "Can you tell the {attribute_type} of {name} located at region {bbox}?",
			"response": "{attribute_values}"
		},
		{
			"prompt"  : "Identify the {attribute_type} of the {name} at region {bbox}.",
			"response": "{attribute_values}"
		},
		{
			"prompt"  : "List the {attribute_type} of {name} found at region {bbox}.",
			"response": "{attribute_values}"
		},
		{
			"prompt"  : "Give the {attribute_type} of {name} present at region {bbox}.",
			"response": "{attribute_values}"
		},
		{
			"prompt"  : "Tell me the {attribute_type} for the {name} positioned at region {bbox}.",
			"response": "{attribute_values}"
		}
	]


## relation

def ExistsRelationGenerator_template():
	return [
		{
			"prompt"  : "What is the relation between {object1} and {object2}?",
			"response": "{relation}"
		},
		{
			"prompt"  : "Describe the relation between {object1} and {object2}.",
			"response": "{relation}"
		},
		{
			"prompt"  : "How are {object1} and {object2} related?",
			"response": "{relation}"
		},
		{
			"prompt"  : "What relation exists between {object1} and {object2}?",
			"response": "{relation}"
		},
		{
			"prompt"  : "Can you identify the relation between {object1} and {object2}?",
			"response": "{relation}"
		},
		{
			"prompt"  : "What is the connection between {object1} and {object2}?",
			"response": "{relation}"
		},
		{
			"prompt"  : "Specify the relation between {object1} and {object2}.",
			"response": "{relation}"
		},
		{
			"prompt"  : "Explain the relationship between {object1} and {object2}.",
			"response": "{relation}"
		},
		{
			"prompt"  : "What kind of relation exists between {object1} and {object2}?",
			"response": "{relation}"
		},
		{
			"prompt"  : "What is the specific relation between {object1} and {object2}?",
			"response": "{relation}"
		}
	]


def RelationBBoxGenerator_template():
	return [
		{
			"prompt"  : "What is the relation between objects at {bbox1} and {bbox2}?",
			"response": "{relation}"
		},
		{
			"prompt"  : "Describe the relation between the objects at {bbox1} and {bbox2}.",
			"response": "{relation}"
		},
		{
			"prompt"  : "How are the objects at {bbox1} and {bbox2} related?",
			"response": "{relation}"
		},
		{
			"prompt"  : "What is the relationship between the objects at {bbox1} and {bbox2}?",
			"response": "{relation}"
		},
		{
			"prompt"  : "Identify the relation between objects at {bbox1} and {bbox2}.",
			"response": "{relation}"
		},
		{
			"prompt"  : "What connection exists between objects at {bbox1} and {bbox2}?",
			"response": "{relation}"
		},
		{
			"prompt"  : "Specify the relation between objects at {bbox1} and {bbox2}.",
			"response": "{relation}"
		},
		{
			"prompt"  : "Explain the relationship between objects at {bbox1} and {bbox2}.",
			"response": "{relation}"
		},
		{
			"prompt"  : "What kind of relation exists between objects at {bbox1} and {bbox2}?",
			"response": "{relation}"
		},
		{
			"prompt"  : "What is the specific relation between objects at {bbox1} and {bbox2}?",
			"response": "{relation}"
		}
	]


def HeadRelationGenerator_template():
	return [
		{
			"prompt"  : "Among {candidates}, what is {relation} {object2}?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Out of {candidates}, which is {relation} {object2}?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Which of {candidates} is {relation} {object2}?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Identify which among {candidates} is {relation} {object2}.",
			"response": "{object1}"
		},
		{
			"prompt"  : "Among {candidates}, which one is {relation} {object2}?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Who among {candidates} is {relation} {object2}?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Out of {candidates}, who is {relation} {object2}?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Among {candidates}, who is {relation} {object2}?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Identify the one who is {relation} {object2} among {candidates}.",
			"response": "{object1}"
		},
		{
			"prompt"  : "Who is {relation} {object2} in the group of {candidates}?",
			"response": "{object1}"
		}
	]


## object depths

def CloserObjectGenerator_template():
	return [
		{
			"prompt"  : "Which of {candidates} is closer?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Identify the closer object between {candidates}.",
			"response": "{object1}"
		},
		{
			"prompt"  : "Out of {candidates}, which object is nearer?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Between {candidates}, which object is closer?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Which is the closest object among {candidates}?",
			"response": "{object1}"
		},
		{
			"prompt"  : "From {candidates}, which object is closer to the viewer?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Which object is nearest among {candidates}?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Determine the closer object from {candidates}.",
			"response": "{object1}"
		},
		{
			"prompt"  : "Of {candidates}, which object is closer?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Which object in {candidates} is closest?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Which of the following objects is closer: {candidates}?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Among {candidates}, which object is the nearest?",
			"response": "{object1}"
		}
	]


def FartherObjectGenerator_template():
	return [
		{
			"prompt"  : "Which of {candidates} is farther?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Identify the farther object between {candidates}.",
			"response": "{object1}"
		},
		{
			"prompt"  : "Out of {candidates}, which object is farther?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Between {candidates}, which object is farther?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Which is the farthest object among {candidates}?",
			"response": "{object1}"
		},
		{
			"prompt"  : "From {candidates}, which object is farther away?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Which object is furthest among {candidates}?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Determine the farther object from {candidates}.",
			"response": "{object1}"
		},
		{
			"prompt"  : "Of {candidates}, which object is farther?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Which object in {candidates} is farthest?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Which of the following objects is farther: {candidates}?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Among {candidates}, which object is the farthest?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Out of the given {candidates}, which one is farther?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Can you tell which object is farther from {candidates}?",
			"response": "{object1}"
		},
		{
			"prompt"  : "Which object, among {candidates}, is located farther away?",
			"response": "{object1}"
		}
	]


def CompareObjectDistanceGenerator_template():
	return [
		{
			"prompt"  : "Which of {candidates} is closer to the {anchor}?",
			"response": "{closer}"
		},
		{
			"prompt"  : "Which of {candidates} is farther to the {anchor}?",
			"response": "{farther}"
		},
	]


## Depth

def CloserPointGenerator_template():
	return [
		{
			"prompt"  : "Which point of {candidates} is closer?",
			"response": "{point1}"
		},
		{
			"prompt"  : "Which point among {candidates} is closer?",
			"response": "{point1}"
		},
		{
			"prompt"  : "Identify the closer point between {candidates}.",
			"response": "{point1}"
		},
		{
			"prompt"  : "Out of {candidates}, which point is nearer?",
			"response": "{point1}"
		},
		{
			"prompt"  : "Between {candidates}, which point is closer?",
			"response": "{point1}"
		},
		{
			"prompt"  : "Which is the closest point among {candidates}?",
			"response": "{point1}"
		},
		{
			"prompt"  : "From {candidates}, which point is closer to the origin?",
			"response": "{point1}"
		},
		{
			"prompt"  : "Which point is nearest among {candidates}?",
			"response": "{point1}"
		},
		{
			"prompt"  : "Determine the closer point from {candidates}.",
			"response": "{point1}"
		},
		{
			"prompt"  : "Of {candidates}, which point is closer?",
			"response": "{point1}"
		},
		{
			"prompt"  : "Which point in {candidates} is closest?",
			"response": "{point1}"
		},
	]


def FartherPointGenerator_template():
	return [
		{
			"prompt"  : "Which point of {candidates} is farther?",
			"response": "{point1}"
		},
		{
			"prompt"  : "Which point among {candidates} is farther?",
			"response": "{point1}"
		},
		{
			"prompt"  : "Identify the farther point between {candidates}.",
			"response": "{point1}"
		},
		{
			"prompt"  : "Out of {candidates}, which point is farther?",
			"response": "{point1}"
		},
		{
			"prompt"  : "Between {candidates}, which point is farther?",
			"response": "{point1}"
		},
		{
			"prompt"  : "Which is the farthest point among {candidates}?",
			"response": "{point1}"
		},
		{
			"prompt"  : "From {candidates}, which point is farther away?",
			"response": "{point1}"
		},
		{
			"prompt"  : "Which point is furthest among {candidates}?",
			"response": "{point1}"
		},
		{
			"prompt"  : "Determine the farther point from {candidates}.",
			"response": "{point1}"
		},
		{
			"prompt"  : "Of {candidates}, which point is farther?",
			"response": "{point1}"
		},
		{
			"prompt"  : "Which point in {candidates} is farthest?",
			"response": "{point1}"
		}
	]


## Segment


def ThreePointSegGenerator_template():
	return [
		{
			"prompt"  : "Which point of {candidates} is in the same object with {anchor_point}?",
			"response": "{same_point}"
		},
		{
			"prompt"  : "Which point of {candidates} is not in the same object with {anchor_point}?",
			"response": "{diff_point}"
		},
		{
			"prompt"  : "Out of {candidates}, which point is in the same object as {anchor_point}?",
			"response": "{same_point}"
		},
		{
			"prompt"  : "Among {candidates}, which point is in the same object with {anchor_point}?",
			"response": "{same_point}"
		},
		{
			"prompt"  : "Which point among {candidates} is not in the same object as {anchor_point}?",
			"response": "{diff_point}"
		},
		{
			"prompt"  : "From {candidates}, which point is in the same object as {anchor_point}?",
			"response": "{same_point}"
		},
		{
			"prompt"  : "Identify the point among {candidates} that is in the same object as {anchor_point}.",
			"response": "{same_point}"
		},
		{
			"prompt"  : "Which point from {candidates} is in the same object as {anchor_point}?",
			"response": "{same_point}"
		},
		{
			"prompt"  : "Which of {candidates} is not in the same object as {anchor_point}?",
			"response": "{diff_point}"
		},
		{
			"prompt"  : "Among {candidates}, which point does not belong to the same object as {anchor_point}?",
			"response": "{diff_point}"
		}
	]
