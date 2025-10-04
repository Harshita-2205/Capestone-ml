# src/crew/hugging_face_ner.py
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load tokenizer + model once
MODEL_NAME = "Clinical-AI-Apollo/Medical-NER"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)


def process_ner_output(text: str, max_length: int = 512):
    """
    Run NER on given text using Hugging Face Medical NER model.
    Returns:
        filtered_result: [(token, label)] excluding 'O'
        unique_tags: list of unique entity categories
    """
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model(**encoding)
        logits = output.logits

    # Predict label indices
    predicted_label_indices = torch.argmax(logits, dim=2)
    label_map = model.config.id2label
    predicted_labels = [label_map[idx.item()] for idx in predicted_label_indices[0]]

    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    # Combine and filter
    result = list(zip(tokens, predicted_labels))
    filtered_result = [
        (token.lstrip("▁"), label)
        for token, label in result if label != "O"
    ]

    # Collect unique entity tags
    unique_tags = []
    for entry in predicted_labels:
        if entry.startswith("B") and entry[2:] not in unique_tags:
            unique_tags.append(entry[2:])

    return filtered_result, unique_tags


def generate_clean_ner_report(tagged_tokens, unique_tags):
    """
    Convert NER predictions into structured report format.
    Example:
        {"DISEASE": ["diabetes"], "MEDICATION": ["metformin"]}
    """
    unwanted_tokens = {'[CLS]', '[SEP]'}
    report = {tag: [] for tag in unique_tags if tag != "SEVERITY"}

    current_entity, current_label = [], None

    for token, tag in tagged_tokens:
        token = token.replace('▁', '')
        if token in unwanted_tokens:
            continue

        if tag.startswith("B-"):
            if current_entity:
                report[current_label].append(" ".join(current_entity))
            current_entity = [token]
            current_label = tag[2:]
        elif tag.startswith("I-") and current_label == tag[2:]:
            current_entity.append(token)
        else:
            if current_entity:
                report[current_label].append(" ".join(current_entity))
            current_entity, current_label = [], None

    if current_entity:
        report[current_label].append(" ".join(current_entity))

    return report


# ------------------------------
# Helper for CrewAI integration
# ------------------------------
def run_medical_ner(text: str):
    """
    Main callable function for CrewAI agent.
    Returns: dict with structured entities
    """
    tagged_tokens, unique_tags = process_ner_output(text)
    return generate_clean_ner_report(tagged_tokens, unique_tags)
