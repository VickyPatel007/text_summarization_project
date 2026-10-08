# Text Summarization Using Transformers

A complete Deep Learning project for generating concise summaries from long documents.

## Project objective

Text Summarization focuses on generating concise summaries from longer documents using deep learning models such as Transformers or RNNs. This project implements an abstractive text summarization system using a Transformer-based sequence-to-sequence model.

## Features

- CNN/DailyMail dataset support through Hugging Face Datasets
- Text cleaning and preprocessing
- Transformer-based abstractive summarization
- Training/fine-tuning script
- ROUGE evaluation
- Command-line summarization
- Flask web interface
- CPU/GPU/MPS support
- Saved model/tokenizer support

## Project structure

```text
text_summarization_project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── summarize.py
├── data/
├── models/
└── outputs/
```

## 1. Create virtual environment

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run summarization without training

The project can directly use a pretrained Transformer model.

```bash
python src/summarize.py --text "Artificial intelligence is transforming many industries..."
```

Or:

```bash
python src/summarize.py --file article.txt
```

The default model is:

```text
sshleifer/distilbart-cnn-12-6
```

The first run downloads the model from Hugging Face.

## 4. Run the web application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Paste a long article and click Summarize.

## 5. Fine-tune on CNN/DailyMail

Download and preprocess a small subset first:

```bash
python src/train.py --train-samples 2000 --validation-samples 300
```

For a stronger experiment:

```bash
python src/train.py --train-samples 10000 --validation-samples 1000
```

For a full dataset run, remove the sample limits, but this requires substantial RAM/storage and training time.

## 6. Evaluate the model

```bash
python src/evaluate.py --samples 100
```

If you trained a model and saved it in `models/text-summarizer`, evaluate it with:

```bash
python src/evaluate.py --model-path models/text-summarizer --samples 100
```

## Dataset

CNN/DailyMail contains news articles paired with human-written highlights/summaries. The project loads it using:

```python
load_dataset("cnn_dailymail", "3.0.0")
```

You do not need to manually download the dataset.

## Methodology

1. Load article-summary dataset.
2. Clean and normalize text.
3. Tokenize article and summary.
4. Encode the input using a Transformer tokenizer.
5. Pass the encoded article through the Transformer encoder-decoder architecture.
6. Generate an abstractive summary.
7. Compare generated and reference summaries using ROUGE.
8. Deploy the trained/pretrained model through a Flask web application.

## Algorithms / technologies

- Natural Language Processing
- Deep Learning
- Transformer architecture
- Sequence-to-sequence learning
- Attention mechanism
- Abstractive summarization
- ROUGE evaluation
- Hugging Face Transformers
- PyTorch
- Flask

## Expected output

Input:

```text
A long news article or document...
```

Output:

```text
A short, meaningful summary containing the main information.
```

## Notes for college demonstration

For a classroom demo, use the pretrained model first because it gives results without waiting for model training.

For the experiment report, mention that the system uses a Transformer encoder-decoder architecture and ROUGE metrics for evaluation.

## License

Educational project. Dataset and pretrained model remain subject to their respective licenses.
