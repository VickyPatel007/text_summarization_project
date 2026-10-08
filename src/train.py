import argparse
import sys
from pathlib import Path

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DEFAULT_MODEL,
    FINETUNED_MODEL,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
)
from data_loader import load_cnn_dailymail
from preprocess import tokenize_batch

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Transformer for summarization")
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--validation-samples", type=int, default=300)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    print("Loading CNN/DailyMail...")
    dataset = load_cnn_dailymail(
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(FINETUNED_MODEL / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=False,
        report_to="none",
        load_best_model_at_end=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    FINETUNED_MODEL.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINETUNED_MODEL))
    tokenizer.save_pretrained(str(FINETUNED_MODEL))

    print(f"\nModel saved to: {FINETUNED_MODEL}")

if __name__ == "__main__":
    main()
