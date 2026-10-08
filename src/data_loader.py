from datasets import load_dataset

def load_cnn_dailymail(train_samples=None, validation_samples=None, test_samples=None):
    """
    Load CNN/DailyMail v3.0.0 from Hugging Face.
    Optional sample limits make experimentation much faster.
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    if train_samples:
        dataset["train"] = dataset["train"].select(
            range(min(train_samples, len(dataset["train"])))
        )

    if validation_samples:
        dataset["validation"] = dataset["validation"].select(
            range(min(validation_samples, len(dataset["validation"])))
        )

    if test_samples:
        dataset["test"] = dataset["test"].select(
            range(min(test_samples, len(dataset["test"])))
        )

    return dataset
