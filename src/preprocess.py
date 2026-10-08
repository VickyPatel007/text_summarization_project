import re
from transformers import AutoTokenizer
from config import DEFAULT_MODEL, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH

def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_tokenizer(model_name=DEFAULT_MODEL):
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_batch(batch, tokenizer):
    inputs = [clean_text(x) for x in batch["article"]]
    targets = [clean_text(x) for x in batch["highlights"]]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
