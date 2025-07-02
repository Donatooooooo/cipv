import json, re, random

def clean_text_light(text):
    """
    Applies light cleaning: trims whitespace, removes 
    invisible characters, digits, and lowers text.
    """
    
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u200b-\u200f]", "", text)
    text = re.sub(r"[\d/]+", "", text)
    return text

def format_dialogue(conversation):
    """
    Formats the dialogue turn-by-turn, adding 
    speaker tags and a separator token.
    """

    turns = []
    for idx, turn in enumerate(conversation["dialogue"]):
        speaker = turn["speaker"].upper()
        text = turn["text"].strip()
        turns.append(f"[{speaker}] {text}")
    return clean_text_light(f" ".join(turns))

def format_segments(conversation):
    """
    Return a list of structured segments 
    [HIST][CURR][NEXT]
    """
    
    dialogue = conversation["dialogue"]
    name1 = conversation["name1"]
    name2 = conversation["name2"]

    speaker_map = {name1: "A", name2: "B"}

    num_turns = len(dialogue)
    all_segments = []

    for idx, turn in enumerate(dialogue):
        segments = []

        # [HIST]
        if idx > 0:
            prev = dialogue[idx - 1]
            prev_speaker = speaker_map.get(prev["speaker"], prev["speaker"])
            prev_text = clean_text_light(prev["text"].strip())
            segments.append(f"[HIST] [{prev_speaker}] {prev_text}")

        # [CURR]
        curr_speaker = speaker_map.get(turn["speaker"], turn["speaker"])
        curr_text = clean_text_light(turn["text"].strip())
        segments.append(f"[CURR] [{curr_speaker}] {curr_text}")

        # [NEXT]
        if idx < num_turns - 1:
            nxt = dialogue[idx + 1]
            next_speaker = speaker_map.get(nxt["speaker"], nxt["speaker"])
            next_text = clean_text_light(nxt["text"].strip())
            segments.append(f"[NEXT] [{next_speaker}] {next_text}")

        all_segments.append(" ".join(segments))

    return all_segments

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def example_loader(baseline = False):
    """
    Loads and formats example conversations from a JSON file.
    """
    
    conversations = load_json("./data/processed/examples_generated.json")
    conversation = random.choice(conversations)
    if baseline:
        texts = " ".join(turn["text"] for turn in conversation["dialogue"])
    else:
        texts = format_segments(conversations)
        texts = [" ".join(t) if isinstance(t, list) else t for t in texts]
    return texts, conversation["dialogue"], conversation["person_couple"]