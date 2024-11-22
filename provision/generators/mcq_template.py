def make_options(choices, format='letter'):
	assert format in ['numeric', 'letter']
	if format == 'numeric':
		prefix = [str(i + 1) for i in range(len(choices))]
	else:
		prefix = [chr(ord("a") + i).upper() for i in range(len(choices))]
	options1 = [f"{p}. {c}" for p, c in zip(prefix, choices)]
	options2 = [f"({p}) {c}" for p, c in zip(prefix, choices)]
	return prefix, options1, options2


CHOICE_PATTERN = [
	"",
	"Choices: ",
	"Options: ",
	"Selections: ",
	"Pick from: ",
	"Select from: ",
	"Choose from: ",
]
CHOICE_PATTERN = [f"\n{p}" for p in CHOICE_PATTERN] + [f" {p}" for p in CHOICE_PATTERN]


def make_multiple_choice_qa(data, candidates, answer, rng):
	assert len(set(candidates)) == len(candidates), "candidates should be unique"
	assert len(candidates), "candidates should not be empty"
	candidates = list(rng.permutation(candidates))
	prefix, options1, options2 = make_options(candidates)
	answer_id = candidates.index(answer)
	options = rng.choice([options1, options2])
	answer = options[answer_id]
	data['prompt'] = data['prompt'] + rng.choice(CHOICE_PATTERN) + ", ".join(options)
	data['response'] = answer
	return data
