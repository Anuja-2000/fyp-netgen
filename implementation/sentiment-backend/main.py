from flask import Flask, request, jsonify
import pandas as pd
import torch
import os
from utils.preprocess import split_into_sentences
from utils.preprocess import clean_text
from utils.aspect_model import load_aspect_model
from utils.sentiment_model import load_sentiment_model
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils.preprocess import split_into_sentences
from utils.safety import predict_safety
from flask_cors import CORS
from utils.deberta import load_deberta_model
 
app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ASPECT_LABELS = [
    "Wellness & Relaxation",
    "Transportation",
    "Food & Dining",
    "Nature & Activities",
    "Entertainment & Shopping",
    "Accommodation",
    "Crowds & Sustainability",
    "Religious & Historical"
]
SENTIMENT_LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]

CSV_PATH = "dataset/Reviews_SriLankan_destinations-sentences_labeled_sentiment_by_aspect_type_safety_correct.csv"
SUMMARY_CSV_PATH = "dataset/summary.csv"

# Load models
aspect_tokenizer, aspect_model = load_aspect_model("models/aspect_model/multi_label_bert.pth", device)
sentiment_tokenizer, sentiment_model = load_sentiment_model("models/tri_head_saved_model", device)
deberta_model, deberta_tokenizer = load_deberta_model("models/baseline_deberta", device)

# user type mapping
def get_user_type(travel_group):
    travel_group_to_user_type = {
        "Traveling with teenagers (12-18)": "family",
        "Traveling with friends": "group",
        "Traveling with extended family (multi-generational)": "family",
        "Traveling with young kids (under 12)": "family",
        "Traveling with a partner": "couple",
        "Solo traveler": "solo"
    }
    return travel_group_to_user_type.get(travel_group, "unknown")

# Ensure summary file exists
def initialize_summary_csv():
    if not os.path.exists(SUMMARY_CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        update_summary_csv(df)

# Aspect detection
def predict_aspects(text):
    encoded = aspect_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        outputs = aspect_model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs).cpu()[0]
    return [ASPECT_LABELS[i] for i, p in enumerate(probs) if p >= 0.5]

# Sentiment detection
def predict_sentiment(text, aspect):
    input_text = f"[ASPECT] {aspect} [SENTENCE] {text}"
    encoded = sentiment_tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        outputs = sentiment_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        pred = torch.argmax(logits, dim=1).item()
    return SENTIMENT_LABELS[pred]


def predict_sentiment_deberta(sentence, aspect, model, tokenizer, device):
    model.eval()  # Set model to evaluation mode
    text = f"[ASPECT] {aspect} [SENTENCE] {sentence}"
    encoded = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        pred_label_id = torch.argmax(logits, dim=1).item()

    return SENTIMENT_LABELS[pred_label_id]    

# Summary update logic
def update_summary_csv(df):
    summaries = []
    for (place, user_type), group in df.groupby(["Location_Name", "user_type"]):
        summary_row = {"place": place, "user_type": user_type}
        total_reviews = len(group)
        unsafe_reviews = group[group["safety_label"] != 0]
        unsafe_percent = round(100 * len(unsafe_reviews) / total_reviews, 2) if total_reviews > 0 else 0.0
        summary_row["unsafe_review_percent"] = unsafe_percent

        for aspect in ASPECT_LABELS:
            if aspect not in group.columns:
                continue
            counts = group[aspect].value_counts()
            total = counts.sum()
            aspect_summary = {
                sentiment: round(100 * counts.get(sentiment, 0) / total, 2)
                for sentiment in SENTIMENT_LABELS
            }
            summary_row[aspect] = aspect_summary

        summaries.append(summary_row)

    flat_rows = []
    for row in summaries:
        flat = {
            "place": row["place"],
            "user_type": row["user_type"],
            "unsafe_review_percent": row["unsafe_review_percent"]
        }
        for aspect in ASPECT_LABELS:
            val = row.get(aspect, {})
            flat[aspect] = "; ".join(f"{k}: {v}%" for k, v in val.items()) if val else ""
        flat_rows.append(flat)

    pd.DataFrame(flat_rows).to_csv(SUMMARY_CSV_PATH, index=False)
#rule base since on CPU    
def generate_summary_from_row(row):
    place = row.get("place", "Unknown")
    user_type = row.get("user_type", "unknown")
    unsafe = row.get("unsafe_review_percent", 0.0)

    summary_lines = [f"{place} is a travel destination reviewed by {user_type} visitors."]

    for aspect in ASPECT_LABELS:
        sentiment_data = row.get(aspect, "")
        if not sentiment_data:
            continue
        try:
            sentiment_parts = sentiment_data.split("; ")
            sentiment_dict = {k.strip(): float(v.strip("%")) for k, v in (s.split(": ") for s in sentiment_parts)}

            top_sentiments = sorted(sentiment_dict.items(), key=lambda x: x[1], reverse=True)[:2]
            if top_sentiments[0][1] == 0:
                continue

            sentiment_phrase = ", ".join(f"{k} ({v}%)" for k, v in top_sentiments)
            summary_lines.append(f"For {aspect.lower()}, top sentiments were: {sentiment_phrase}.")
        except:
            continue

    if unsafe > 5:
        summary_lines.append(f"Note: {unsafe}% of reviews mentioned safety concerns.")
    else:
        summary_lines.append(f"Safety concerns were minimal ({unsafe}%).")

    return " ".join(summary_lines)

'''
# Generate natural language summary with T5
def generate_summary_from_row(row):
    place = row.get("place", "Unknown")
    user_type = row.get("user_type", "unknown")
    unsafe = row.get("unsafe_review_percent", 0.0)

    summary_intro = f"{place} is a destination reviewed by {user_type} travelers. Here's what they felt across different aspects:"

    aspect_summaries = []

    for aspect in ASPECT_LABELS:
        sentiment_data = row.get(aspect, "")
        if not sentiment_data:
            continue

        try:
            sentiment_parts = sentiment_data.split("; ")
            sentiment_dict = {k.strip(): float(v.strip("%")) for k, v in (s.split(": ") for s in sentiment_parts)}
            items = [f"{k} ({v}%)" for k, v in sentiment_dict.items() if v > 0]
            aspect_sentence = f"- {aspect}: " + ", ".join(items) + "."
            aspect_summaries.append(aspect_sentence)

        except Exception:
            aspect_summaries.append(f"- {aspect}: data not available.")

    safety_note = f"Safety concerns were {'minimal' if unsafe <= 5 else f'reported by {unsafe}% of users'}."

    # Combine all input for T5
    full_text = summary_intro + "\n" + "\n".join(aspect_summaries) + "\n" + safety_note

    # Generate summary
    input_ids = t5_tokenizer.encode("summarize: " + full_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = t5_model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    '''


# Predict endpoint
@app.route("/predict_review", methods=["POST"])
def predict_review():
    data = request.json
    review = data.get("review", "")
    user_type = data.get("user_type", "unknown")
    location_name = data.get("location_name", "Unknown")

    sentences = split_into_sentences(review)

    rows = []
    safety_flags = []
    sentence_results = []

    for sentence in sentences:
        aspects = predict_aspects(sentence)
        sentiments = {aspect: predict_sentiment(sentence, aspect) for aspect in aspects}
        safety_pred = predict_safety(sentence)
        safety_label = 1 if safety_pred["label"] == "Unsafe" else 0
        safety_flags.append(safety_label)

        row = {
            "Location_Name": location_name,
            "user_type": user_type,
            "Text": review,
            "review sentences": sentence,
            "safety_label": safety_label,
            "safety_keyword": safety_pred["label"].lower(),
            **{aspect: sentiments.get(aspect, "") for aspect in ASPECT_LABELS}
        }
        rows.append(row)

        # Build per-sentence result
        sentence_results.append({
            "sentence": sentence,
            "safety_label": safety_label,
            "safety_keyword": safety_pred["label"].lower(),
            "aspects": sentiments
        })

    # Append to CSV
    df = pd.read_csv(CSV_PATH)
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

    update_summary_csv(df)

    return jsonify({
        "num_sentences": len(sentences),
        "sentence_results": sentence_results,
        "user_type": user_type,
        
    })


@app.route("/predict_review_deberta", methods=["POST"])
def predict_review_deberta():
    data = request.json
    review = data.get("review", "")
    user_type = data.get("user_type", "unknown")
    location_name = data.get("location_name", "Unknown")

    if not review:
        return jsonify({"error": "Review text is required."}), 400

    sentences = split_into_sentences(review)
    all_results = []

    for sentence in sentences:
        # Predict aspects
        encoded = aspect_tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            aspect_model.eval()
            aspect_logits = aspect_model(input_ids, attention_mask)
            aspect_probs = torch.sigmoid(aspect_logits).cpu()[0]

        predicted_aspects = [ASPECT_LABELS[i] for i, p in enumerate(aspect_probs) if p >= 0.5]

        # Predict sentiment per aspect with DeBERTa
        sentiments = {}
        for aspect in predicted_aspects:
            sentiment = predict_sentiment_deberta(sentence, aspect, deberta_model, deberta_tokenizer, device)
            sentiments[aspect] = sentiment

        all_results.append({
            "sentence": sentence,
            "predicted_aspects": predicted_aspects,
            "sentiments": sentiments
        })

    return jsonify({
        "review": review,
        "user_type": user_type,
        "location_name": location_name,
        "sentence_results": all_results,
        "num_sentences": len(sentences)
    })


# Per-user-type summary API
@app.route("/place_summary", methods=["GET"])
def place_summary():
    place_name = request.args.get("place", "")
    if not place_name:
        return jsonify({"error": "Missing place name"}), 400

    df_summary = pd.read_csv(SUMMARY_CSV_PATH)
    filtered = df_summary[df_summary["place"].str.lower() == place_name.lower()]
    if filtered.empty:
        return jsonify({"message": f"No summary data found for place: {place_name}"}), 404

    natural_summaries = [generate_summary_from_row(row) for _, row in filtered.iterrows()]
    return jsonify({
        "summary_rows": filtered.to_dict(orient="records"),
        "natural_summaries": natural_summaries
    })

@app.route("/place_summary_by_usertype", methods=["POST"])
def place_summary_by_usertype():
    session_data = request.get_json()
    if not session_data:
        return jsonify({"error": "Missing or invalid JSON data"}), 400

    user_profile = session_data.get("user_profile", {})
    travel_group = user_profile.get("Travel Group", "")
    user_type = get_user_type(travel_group)

    recommended_places = session_data.get("recommended_places", [])
    if recommended_places:
        places_to_summarize = recommended_places
    else:
        location = user_profile.get("Location")
        if not location:
            return jsonify({"error": "No recommended_places or Location provided"}), 400
        places_to_summarize = [location]

    df_summary = pd.read_csv(SUMMARY_CSV_PATH)
    results = []

    for place_name in places_to_summarize:
        filtered = df_summary[
            (df_summary["place"].str.lower() == place_name.lower()) &
            (df_summary["user_type"].str.lower() == user_type)
        ]

        if filtered.empty:
            results.append({
                "place": place_name,
                "user_type": user_type,
                "natural_summary": f"No summary data found for place '{place_name}' and user type '{user_type}'."
            })
        else:
            natural_summaries = [generate_summary_from_row(row) for _, row in filtered.iterrows()]
            combined_summary = " ".join(natural_summaries)
            results.append({
                "place": place_name,
                "user_type": user_type,
                "natural_summary": combined_summary,
                "summary_rows": filtered.to_dict(orient="records")
            })

    return jsonify({"results": results})


# Overall summary for all user types
@app.route("/place_summary_overall", methods=["GET"])
def place_summary_overall():
    place_name = request.args.get("place", "")
    if not place_name:
        return jsonify({"error": "Missing place name"}), 400

    df = pd.read_csv(CSV_PATH)
    df_place = df[df["Location_Name"].str.lower() == place_name.lower()]
    if df_place.empty:
        return jsonify({"message": f"No data found for place: {place_name}"}), 404

    total_reviews = len(df_place)
    unsafe_percent = round(100 * len(df_place[df_place["safety_label"] != 0]) / total_reviews, 2)
    summary_row = {"place": place_name, "user_type": "all", "unsafe_review_percent": unsafe_percent}

    for aspect in ASPECT_LABELS:
        if aspect in df_place.columns:
            counts = df_place[aspect].value_counts()
            total = counts.sum()
            sentiment_dist = {
                s: round(100 * counts.get(s, 0) / total, 2)
                for s in SENTIMENT_LABELS
            }
            summary_row[aspect] = "; ".join(f"{k}: {v}%" for k, v in sentiment_dist.items())

    summary = generate_summary_from_row(summary_row)

    return jsonify({
        "summary_row": summary_row,
        "natural_summary": summary
    })

if __name__ == "__main__":
    initialize_summary_csv()
    app.run(debug=True)

