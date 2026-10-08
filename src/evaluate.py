import argparse
import sys
from pathlib import Path

import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFAULT_MODEL, FINETUNED_MODEL, DEVICE
from data_loader import load_cnn_dailymail

def evaluate_model(model_path, samples=100):
    dataset = load_cnn_dailymail(test_samples=samples)["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
    model.eval()

    predictions = []
    references = []

    for row in dataset:
        inputs = tokenizer(
            row["article"],
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(DEVICE)

        output_ids = model.generate(
            **inputs,
            max_length=128,
            min_length=20,
            num_beams=4,
            no_repeat_ngram_size=3,
        )

        prediction = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        predictions.append(prediction)
        references.append(row["highlights"])

    rouge = evaluate.load("rouge")
    scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )

    print("\nROUGE Evaluation Results")
    print("-" * 30)
    for name, value in scores.items():
        print(f"{name}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    model_path = args.model_path or (
        str(FINETUNED_MODEL) if FINETUNED_MODEL.exists() else DEFAULT_MODEL
    )

    evaluate_model(model_path, args.samples)
