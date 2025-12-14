import re
import math
from transformers import pipeline
from langdetect import detect, LangDetectException

sentiment_pipeline = pipeline("sentiment-analysis")
emotion_pipeline = pipeline("text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True)


# ---------- Language Detection ----------
def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


# ---------- Entropy (Gibberish Detection) ----------
def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0

    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1

    entropy = 0.0
    length = len(text)

    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)

    return entropy


def looks_like_gibberish(text: str) -> bool:
    text = text.lower().strip()
    compact = re.sub(r"\s+", "", text)

    # High randomness
    if shannon_entropy(compact) > 4.2:
        return True

    # Vowel ratio
    letters = sum(c.isalpha() for c in compact)
    vowels = sum(c in "aeiou" for c in compact)

    if letters > 0 and vowels / letters < 0.25:
        return True

    # No real word pattern
    if not re.search(r"[a-z]{3,}", text):
        return True

    return False


# ---------- Validation ----------
def is_invalid_text(text: str) -> bool:
    if not text or text.strip() == "":
        return True

    if not re.search(r"[a-zA-Z]", text):
        return True

    if len(text.strip()) < 3:
        return True

    non_letters = re.sub(r"[a-zA-Z\s]", "", text)
    if len(non_letters) / len(text) > 0.5:
        return True

    if looks_like_gibberish(text):
        return True

    if not is_english(text):
        return True

    return False


# ---------- Sentiment Analyzer ----------
def sentiment_analyzer(text_to_analyse):
    if is_invalid_text(text_to_analyse):
        return {
            "label": None,
            "score": None,
            "error": "Invalid, gibberish, or non-English text"
        }

    result = sentiment_pipeline(text_to_analyse)[0]

    if result["score"] < 0.6:
        return {
            "label": None,
            "score": None,
            "error": "Low confidence prediction"
        }

    return {
        "label": result["label"].lower(),
        "score": round(result["score"], 4)
    }

def emotion_analyzer(text_to_analyse):
    if is_invalid_text(text_to_analyse):
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "surprise": None,
            "neutral": None,
            "dominant_emotion": None,
            "error": "Invalid, gibberish, or non-English text"
        }

    predictions = emotion_pipeline(text_to_analyse)[0]

    scores = {p["label"]: round(p["score"], 4) for p in predictions}
    dominant_emotion = max(scores, key=scores.get)

    # Confidence check on dominant emotion
    if scores[dominant_emotion] < 0.6:
        return {
            **scores,
            "dominant_emotion": None,
            "error": "Low confidence prediction"
        }

    return {
        **scores,
        "dominant_emotion": dominant_emotion
    }





