from flask import Flask, render_template, request
from SentimentAnalysis.sentiment_analysis import sentiment_analyzer, emotion_analyzer

app = Flask("Sentiment Analyzer")

@app.route("/sentimentAnalyzer")
def sent_analyzer():
    text_to_analyze = request.args.get("textToAnalyze")

    # Sentiment
    sentiment_response = sentiment_analyzer(text_to_analyze)

    if sentiment_response.get("label") is None:
        return "Invalid input! Try again."

    label = sentiment_response["label"]
    score = sentiment_response["score"]

    # Emotion
    emotion_response = emotion_analyzer(text_to_analyze)

    if emotion_response.get("dominant_emotion") is None:
        return "Invalid input! Try again."

    anger = emotion_response.get("anger")
    disgust = emotion_response.get("disgust")
    fear = emotion_response.get("fear")
    joy = emotion_response.get("joy")
    sadness = emotion_response.get("sadness")
    surprise = emotion_response.get("surprise")
    neutral = emotion_response.get("neutral")
    dominant_emotion = emotion_response.get("dominant_emotion")

    return (
        f"The given text has been identified as <b>{label}</b> "
        f"with a score of <b>{score}</b>.<br><br>"
        f"<b>Emotions:</b><br>"
        f"Anger: {anger}<br>"
        f"Disgust: {disgust}<br>"
        f"Fear: {fear}<br>"
        f"Joy: {joy}<br>"
        f"Sadness: {sadness}<br>"
        f"Surprise: {surprise}<br>"
        f"Neutral: {neutral}<br><br>"
        f"<b>Dominant Emotion:</b> {dominant_emotion}"
    )


@app.route("/")
def render_index_page():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5016, debug=True)
