import argparse
from pathlib import Path
import sys

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DEFAULT_MODEL,
    FINETUNED_MODEL,
    DEVICE,
    MIN_SUMMARY_LENGTH,
    MAX_SUMMARY_LENGTH,
)

def choose_model(model_path=None):
    if model_path:
        return str(model_path)
    if FINETUNED_MODEL.exists():
        return str(FINETUNED_MODEL)
    return DEFAULT_MODEL

def summarize_text(text, model_path=None):
    text = text.strip()
    if not text:
        raise ValueError("Input text cannot be empty.")

    model_name = choose_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model = model.to(DEVICE)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(DEVICE)

    output_ids = model.generate(
        **inputs,
        max_length=MAX_SUMMARY_LENGTH,
        min_length=MIN_SUMMARY_LENGTH,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Transformer Text Summarizer")
    parser.add_argument("--text", type=str, help="Text to summarize")
    parser.add_argument("--file", type=str, help="Path to a text file")
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        print("Enter text. Press Ctrl+D (macOS/Linux) or Ctrl+Z (Windows) when finished:")
        text = sys.stdin.read()

    print("\nSUMMARY:\n")
    print(summarize_text(text, args.model_path))

if __name__ == "__main__":
    main()
