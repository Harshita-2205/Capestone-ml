from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch


tokenizer = AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER")
model = AutoModelForTokenClassification.from_pretrained("Clinical-AI-Apollo/Medical-NER")


def process_ner_output(text: str, max_length: int = 512) -> list:
    """
    Process text using a pretrained NER model and tokenizer, returning meaningful predictions.
    
    Parameters:
        text (str): Input text to analyze.
        model: Pretrained Hugging Face model for NER.
        tokenizer: Corresponding tokenizer for the model.
        max_length (int): Maximum sequence length for the tokenizer.

    Returns:
        list: A list of tuples in the format (token, label), excluding 'O' labels.
    """
    # Tokenize the text
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Get logits from the model
    output = model(**encoding)
    logits = output.logits
    
    # Predict label indices
    predicted_label_indices = torch.argmax(logits, dim=2)
    
    # Map label indices to label names
    label_map = model.config.id2label
    predicted_labels = [label_map[idx.item()] for idx in predicted_label_indices[0]]
    
    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    
    # Combine tokens and labels
    result = list(zip(tokens, predicted_labels))
    
    # Filter out non-entity tokens ('O' labels)
    filtered_result = [
        (token.lstrip("▁"), label)  # Remove leading underscores from tokens
        for token, label in result if label != "O"
    ]

    unique_tags = []

    for entry in predicted_labels:
        if entry[0]=="B" and entry not in unique_tags:
            unique_tags.append(entry[2:])
    
    return filtered_result,unique_tags

def generate_clean_ner_report(tagged_tokens, unique_tags):
    """
    Generate a clean NER report based on tagged tokens and unique tags.
    
    Parameters:
        tagged_tokens (list): List of tuples with tokens and labels.
        unique_tags (list): List of unique entity tags.

    Returns:
        dict: A report in the format {tag: [relevant details]}.
    """
    # Initialize report with all unique tags except unwanted ones
    unwanted_tokens = {'[CLS]', '[SEP]'}
    report = {tag: [] for tag in unique_tags if tag != "SEVERITY"}

    current_entity = []
    current_label = None

    for token, tag in tagged_tokens:
        # Remove underscores and filter unwanted tokens
        token = token.replace('▁', '')
        if token in unwanted_tokens:
            continue

        if tag.startswith("B-"):
            # Save the previous entity if it exists
            if current_entity:
                report[current_label].append(" ".join(current_entity))
            # Start a new entity
            current_entity = [token]
            current_label = tag[2:]  # Remove the "B-" prefix
        elif tag.startswith("I-") and current_label == tag[2:]:
            # Continue the current entity if the label matches
            current_entity.append(token)
        else:
            # Save the previous entity and reset if the tag doesn't match
            if current_entity:
                report[current_label].append(" ".join(current_entity))
            current_entity = []
            current_label = None

    # Add the last entity if any
    if current_entity:
        report[current_label].append(" ".join(current_entity))

    return report



