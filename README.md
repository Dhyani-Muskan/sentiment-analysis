# Emotion & Sentiment Analyzer
## 🚀 Project Overview

This application accepts text input from a web interface and:

  - Validates input text (empty, gibberish, non‑English, random characters)

  - Detects sentiment (positive / negative / neutral)

  - Detects emotions (anger, disgust, fear, joy, sadness, surprise, neutral)

  - Returns dominant emotion with confidence checks

  - Includes unit testing for reliability

The project is designed with clean architecture, error handling, and ML confidence thresholds in mind.

## 🧠 Technologies & Libraries Used in Backend

  - Python3.11
  - Flask – Web framework
  - unittest – Built‑in Python testing framework
  - Hugging Face Transformers

 
## 📂 Project Structure
```text
project-root/
│
├── SentimentAnalysis/
│   ├── __init__.py
│   └── sentiment_analysis.py   
│
├── templates/
│   └── index.html             
│
├── server.py                   
├── test_sentiment_analysis.py           
├── requirements.txt
└── README.md
```

# 🔍 Core Functionalities Explained
1️⃣ Language Detection
```text
def is_english(text: str) -> bool:
    return detect(text) == "en"
```
Why? \
Prevents unsupported languages and avoids misleading ML predictions

2️⃣ Gibberish Detection
The app uses Shannon Entropy + Linguistic Rules to reject random or meaningless input.

✔ Entropy Check
```text
if shannon_entropy(compact) > 4.2:
    return True
```
High entropy → random characters

✔ Vowel Ratio
```
vowels / letters < 0.25
```
Human language has predictable vowel usage.

✔ Word Pattern Validation
```
re.search(r"[a-z]{3,}", text)
```
Ensures real words exist.

3️⃣ Input Validation Pipeline
```
def is_invalid_text(text: str) -> bool:
```
Checks:

Empty input, Only symbols or numbers, Very short input, Excessive non‑letters, Gibberish, Non‑English text

This prevents: 500 server errors, ML hallucinations, Bad UX

#😊 Sentiment Analysis
```
def sentiment_analyzer(text_to_analyse):
Output
{
  "label": "positive",
  "score": 0.9876
}
```
Confidence Filtering
```
if result["score"] < 0.6:
```
Low‑confidence predictions are rejected.

#😡 Emotion Detection
```
def emotion_analyzer(text_to_analyse):
Supported Emotions

anger

disgust

fear

joy

sadness

surprise

neutral
```
Dominant Emotion Logic
```
dominant_emotion = max(scores, key=scores.get)
```
Only returned if confidence ≥ 0.6







