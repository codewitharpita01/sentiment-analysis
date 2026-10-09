# SentimentAI — Customer Review Intelligence

A portfolio-ready, local sentiment analysis dashboard for customer feedback. SentimentAI classifies reviews as **Positive**, **Negative**, or **Neutral** using NLTK text normalization, scikit-learn TF-IDF features, and Logistic Regression. Streamlit provides the interface; Plotly powers the interactive charts. No paid API or internet connection is needed at runtime.

> **Dataset note:** `data/sample_reviews.csv` is a small, hand-written teaching dataset included to make the workflow runnable. It is not representative research data and its evaluation scores are only a demonstration of the evaluation process. For meaningful results, obtain a larger dataset with a compatible license and document its source and terms.

## Features

- Analyze one or many pasted reviews, or upload a CSV with a `text` column.
- Review predictions with confidence probabilities and export results as CSV.
- Session-based dashboard metrics, sentiment filters, interactive charts, and common terms from analyzed reviews.
- Inspect the trained model's real held-out accuracy, per-class precision, recall, F1, and confusion matrix.
- Train and run locally. The application does not call a paid LLM or hosted sentiment API.
- Keep negation cues (`not`, `never`, `no`) during preprocessing.

## Screenshots

Add screenshots after running the application, for example:

```text
screenshots/
├── home-dashboard.png
├── analyze-reviews.png
└── visualizations.png
```

## Project structure

```text
sentiment-analysis/
├── app.py                    # Streamlit UI
├── train_model.py            # Data validation, split, training, evaluation
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── sample_reviews.csv    # Small example only
├── models/                   # Generated model (ignored by Git)
└── src/
    ├── __init__.py
    ├── preprocessing.py      # NLTK normalization and stemming
    ├── prediction.py         # Model loading and inference
    └── evaluation.py         # Held-out metrics
```

## Architecture and workflow

1. The trainer reads `text` and `sentiment`, removes missing, blank, duplicate, and unsupported-label rows, then checks that at least two classes remain.
2. It separates a held-out test set before fitting. Stratified sampling is used when each class has enough examples; tiny datasets fall back to an unstratified split.
3. A scikit-learn pipeline fits TF-IDF (unigrams and bigrams) and Logistic Regression on training rows only. The NLTK Snowball stemmer is invoked locally; no NLTK corpus download is needed.
4. The model predicts the held-out rows. Accuracy, per-class precision/recall/F1, and a confusion matrix are calculated from those predictions and stored with the model.
5. Streamlit loads that artifact and uses the same fitted pipeline for pasted text and CSV uploads. Charts use predictions analyzed in the current browser session.

## Dataset format and licensing

Use a UTF-8 comma-separated CSV containing:

```csv
text,sentiment
"The delivery was fast and the product works perfectly",Positive
"It stopped working after a day",Negative
"I received the blue version",Neutral
```

Supported labels are case-insensitive `Positive`, `Negative`, and `Neutral`. Invalid labels and rows without text are dropped. The sample file is provided as project-authored demonstration content with no external dataset license. For a real application, choose a dataset whose license permits your intended use, retain its attribution/license, and document the source. One option to investigate is the public UCI Sentiment Labelled Sentences collection; verify its current license and suitability on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) before use. Convert its labels to the three supported names as appropriate; it may not include neutral examples.

To use your own dataset, save it as `data/reviews.csv`, then run `python train_model.py --data data/reviews.csv`. Add at least two examples per class when you want a stratified split; ensure the test set can contain examples from each class. A class absent from training cannot be predicted by the model. The interface still reports the supported label set, but actual predictions only come from classes learned during training.

## Installation and use

Requirements: Python 3.10–3.12 recommended.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). The first model training is required before launching predictions. To train with a different CSV, pass `--data`; to choose another artifact destination, pass `--model`.

## Evaluation methodology

Training uses a fixed `random_state=42` holdout split. Stratification is enabled where possible; the vectorizer is inside the pipeline, so it learns its vocabulary only from training text. `train_model.py` prints actual test-set metrics and saves them to `models/sentiment_pipeline.joblib`; the About page reads those saved metrics. Never compare or present the included tiny sample's score as a general model benchmark. For credible evaluation, use a sufficiently large, representative labeled dataset, inspect class balance, and consider cross-validation and a separate final test set.

The confidence column is the largest Logistic Regression `predict_proba` value. It is a model probability, not a promise or necessarily a calibrated probability of correctness. The model is a simple bag-of-words baseline: sarcasm, context, domain-specific terms, multilingual text, and mixed sentiment may be difficult.

## Troubleshooting

- **Model not found:** run `python train_model.py` from the project folder.
- **Missing `text`/`sentiment`:** the training file needs both exact column names. An analysis upload only needs `text`.
- **Only one class after cleanup:** provide valid examples for at least two of the supported labels.
- **Streamlit command not found:** activate the virtual environment and run `python -m streamlit run app.py`.
- **CSV parse error:** save a valid UTF-8 CSV; quote review text that contains commas or line breaks.
- **PowerShell blocks activation:** use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or invoke `.venv\Scripts\python.exe` directly.

## GitHub publishing

1. Review the source, sample data, and license/source notes. Keep any real customer data out of the repository.
2. The generated `models/*.joblib`, virtual environments, cache files, and secrets are excluded by `.gitignore`. Users can recreate the model with `python train_model.py`.
3. Initialize and publish from the project folder:

```bash
git init
git add app.py train_model.py src data/sample_reviews.csv requirements.txt README.md .gitignore
git commit -m "Build SentimentAI review intelligence dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sentimentai.git
git push -u origin main
```

Replace the remote with your own repository URL. Add real application screenshots to the `screenshots/` folder and update the screenshot section.

## Future improvements

- Evaluate on a larger licensed, domain-relevant dataset and add robust cross-validation.
- Add multilingual and aspect-based sentiment support.
- Calibrate probabilities and provide model version/data provenance tracking.
- Add configurable phrase-level explanations and deployment guidance.
