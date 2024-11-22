REF_WAY = [
    'There are <region> in the image.',
    'There are some regions <region>.',
    'You can see <region> in the image.',
    'The image shows <region>.',
    'In the image, there are <region>.',
    'The picture contains <region>.',
    'This image displays <region>.',
    'Among the regions, there are <region>.',
    'The image features <region>.',
    'In this picture, you can find <region>.',
    'The image includes <region>.',
    'There are visible <region> in the image.'
]

REF_WAY_NUM = [
    'There are {} regions in the image: <region>.\n',
    'There are {} part regions in the image, given <region>.\n',
    'The image contains {} regions, including <region>.\n',
    'In the image, there are {} regions, such as <region>.\n',
    'This image displays {} regions, including <region>.\n',
    'Among the {} regions in the image, there is <region>.\n',
    'The picture has {} regions, one of which is <region>.\n',
    'You can see {} regions in the image, like <region>.\n',
    'There are {} distinct regions in the image, such as <region>.\n',
    'The image features {} regions, including <region>.\n',
]

MC_QUESTIONS = [
    "Answer with the option's letter from the given choices directly.",
    # "Select the correct option by its letter from the choices provided.",
    # "Choose the answer by its corresponding letter from the options.",
    # "Respond with the letter of the correct choice from the given options.",
    # "Pick the correct option by its letter from the list of choices.",
    # "Answer by selecting the letter of the correct option from the choices.",
    # "Indicate your answer by the letter of the chosen option from the provided choices.",
    # "Choose the correct answer by its letter from the available options.",
    # "Respond with the letter corresponding to the correct choice from the options.",
    # "Select your answer by the letter of the correct option from the given choices.",
    # "Answer with the letter representing the correct option from the provided choices."
]

OBJECT_ATTRIBUTE_QUESTIONS = [
    'Annotate object and attributes for given regions.',
    'How can you label objects and their attributes in specified regions?',
    'Can you identify and describe objects within these regions?',
    'What are the names and characteristics of objects in these areas?',
    'Could you assign names and properties to objects found in the regions?',
    'Please detail the objects and their qualities in the given regions.',
    'How would you categorize objects and their features in these regions?',
    'Can you map objects to their attributes in the specified regions?',
    'Would you outline objects and their attributes in these areas?',
    'How to annotate objects with their names and qualities in the regions?',
    'Could you detail the objects by naming and describing their attributes in these regions?',
    'Can you enumerate the objects and their specific attributes in these regions?',
    'How can objects in these regions be classified and described?',
    'What objects are present in these regions and what are their attributes?',
    'Could you provide names and descriptions for objects in these regions?',
    'Please identify objects and list their attributes in the given regions.',
    'How would you document objects and their characteristics in these areas?',
]

''' Dense Relationship Questions '''
RELATION_QUESTIONS = [
    'Generate list of relationships for: {}.',
    'Assign relations for: {}',
    'Can you assign relations to objects in {}?',
    'How can we map all relationships for {}?',
    'What are the all connections you see in {}?',
    'Identify the inter-regional relationships for {}.',
    'Could you detail the interactions for {}?',
    'Please outline the network of relationships for {}.',
    'Can you delineate the ties binding {}?',
    'Could you classify the types of relationships present in {}?',
]

RELATION_DESCRIPTION_QUESTIONS = [
    'Generate a description for: {}.',
    'Describe {} in details.',
    'What is going on in {}?',
]

RELATION_SUMMARY_QUESTIONS = [
    'Provide a dense summary including all details in the image.',
    'Provide a dense localized narrative of this entire image.',
    'Generate a comprehensive summary of the image.',
]

''' Scene Graph Questions '''
REGION_PROPOSAL_QUESTIONS = [
    'Provide a list of regions to generate scene graph.',
    'Provide a list of bounding boxes to generate scene graph.',
    'List the specific areas within the image for constructing the scene graph.',
    'Identify key regions within the image for scene graph generation.',
]

SG_QUESTIONS = [
    'Generate scene graph for given regions.',
    'Annotate object, attributes, and relations for given regions.',
    'Tag regions with details on objects, their attributes, and relationships.'
    'Map regions to object labels, attributes, and their relations.',
    'Label region contents with names, qualities, and relationships.',
    'Detail regions by object names, attributes, and interactions.',
    'Assign names, traits, and relations to objects in regions.',
    'Annotate regions with object identifiers, characteristics, and links.',
    'Mark regions with object labels, properties, and their relationships.',
    'Define regions by object categories, attributes, and their interrelations.',
    'Can you map regions to object labels, attributes, and their relations?',
]

SG_DETECTION_QUESTIONS = [
    'Generate scene graph with proposed regions in bbox coordinates.',
    'Annotate object, attributes, and relations for given regions with bbox coordinates.',
    'Generate scene graph with bbox coordinates for object, attributes, and relations.',
    'Come up with region proposals with bbox coordinates and generate scene graph for the regions.',
]

SG_COT_QUESTIONS = [
    "Create a scene graph with bbox coordinates to correctly answer the question, then choose the letter of the right option directly.",
    "Use the image to form a scene graph with bbox coordinates that helps in accurately answering the question, and directly pick the corresponding letter of the correct choice.",
    "From the image, generate a scene graph with bbox coordinates to answer the question precisely, and then directly select the appropriate option's letter.",
    "Construct a scene graph with bbox coordinates from the image for an accurate answer, and then directly indicate the correct choice by its letter.",
    "Develop a scene graph with bbox coordinates using the image to ensure the correct answer, and then directly choose the letter of the correct option."
]

''' Grounding Questions '''
GROUNDING_QUESTIONS = [
    'Please provide the region this sentence describes: ',
    'What is the region that best describe this sentence: ',
    'Which area does this sentence correspond to: ',
    'Can you identify the region depicted by this sentence: ',
    'What region is being referred to in this sentence: ',
    'Identify the region described in this sentence: ',
    'What is the region that this sentence describes: '
]

REGION_DESCRIPTION_QUESTIONS = [
    'Give me a short description of <region>.',
    'Can you give me a short description of <region>?',
    'Can you provide me with a short description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in few words?",
    'What can you tell me about the region indicated by <region> in the image in few words?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a concise description?",
    'Could you describe the region shown as <region> in the picture concisely?',
    'What can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a brief description of the region marked with <region> in the image.',
    'Can you give me a brief introduction of the region labeled as <region> in the picture?',
    "I'm interested in knowing the region represented by <region> in the photo. Can you describe it in several words?",
    'What is the region outlined by <region> in the picture like? Could you give me a streamlined description?',
    'Can you provide me with a brief description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in few words, please?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a simple description?",
    'Could you describe the region shown as <region> in the picture in several words?',
    'Please provide me with a simple description of the region marked with <region> in the image, please.',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in few words, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a simple and clear description?',
    'Please describe the region <region> in the image concisely.',
    'Can you offer a simple analysis of the region <region> in the image?',
    'Could tell me something about the region highlighted by <region> in the picture briefly?',
    'Can you share a simple rundown of the region denoted by <region> in the presented image?'
]

GROUNDED_DESCRIPTION_QUESTIONS =  [
    "Can you provide a short description of the image and include the bboxes for each mentioned object?",
    "Please briefly explain what's happening in the photo and give coordinates for the items you reference.",
    "Can you describe the image and provide bounding box coordinates for the objects you mention?",
]

GROUNDED_QA =  [
    "Answer the question by providing the bounding box coordinates for the objects mentioned.",
    "Please provide the bounding box coordinates for the objects you mention in your answer.",
    "Give the bounding box coordinates for the objects you mention in your response.",
]


