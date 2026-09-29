from flask import Flask, send_file, request, jsonify, session, redirect
import pandas as pd
import re
import json
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ========================================
# FLASK APP
# ========================================

app = Flask(__name__)
app.secret_key = "fake-news-classifier-secret-key"


# ========================================
# LOAD DATASET
# ========================================

data = pd.read_csv("dataset.csv")

data["text"] = data["text"].astype(str)

data["label"] = (
    data["label"]
    .astype(str)
    .str.upper()
    .str.strip()
)


print("================================")
print("FAKE NEWS CLASSIFIER")
print("================================")
print("TOTAL NEWS:", len(data))
print("LABEL COUNTS:")
print(data["label"].value_counts())


# ========================================
# TEXT PREPROCESSING
# ========================================

def preprocess_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Keep only alphabets and spaces
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


data["clean_text"] = data["text"].apply(
    preprocess_text
)


# ========================================
# TRAINING DATA
# ========================================

X = data["clean_text"]
y = data["label"]


# ========================================
# TF-IDF FEATURE EXTRACTION
# ========================================

print("================================")
print("TF-IDF FEATURE EXTRACTION")
print("================================")


vectorizer = TfidfVectorizer(
    lowercase=True,
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)


X_features = vectorizer.fit_transform(X)


print(
    "FEATURE MATRIX:",
    X_features.shape
)


# ========================================
# TRAIN TEST SPLIT
# ========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ========================================
# MODEL TRAINING
# ========================================

print("================================")
print("MODEL TRAINING")
print("================================")


model = LogisticRegression(
    max_iter=1000
)


model.fit(
    X_train,
    y_train
)


print("MODEL TRAINING COMPLETED")


# ========================================
# MODEL EVALUATION
# ========================================

print("================================")
print("MODEL EVALUATION")
print("================================")


y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(
    y_test,
    y_pred,
    pos_label="REAL",
    zero_division=0
)


recall = recall_score(
    y_test,
    y_pred,
    pos_label="REAL",
    zero_division=0
)


f1 = f1_score(
    y_test,
    y_pred,
    pos_label="REAL",
    zero_division=0
)


print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)


# ========================================
# CONFIDENCE THRESHOLD
# ========================================

# 0 means UNCERTAIN will not be displayed.
LOW_CONFIDENCE_THRESHOLD = 0.0


# ========================================
# HISTORY
# ========================================

HISTORY_FILE = "history.json"


def load_history():

    if not os.path.exists(HISTORY_FILE):
        return []

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return []


def save_history():

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history_data,
            file,
            indent=4,
            ensure_ascii=False
        )


history_data = load_history()


# ========================================
# HOME
# ========================================

@app.route("/")
def home():

    return send_file(
        "index.html"
    )


# ========================================
# LOGIN
# ========================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        )

        password = request.form.get(
            "password",
            ""
        )

        if username and password:

            session["logged_in"] = True

            session["username"] = username

            return redirect("/check")

        return "Please enter username and password"

    return send_file(
        "login.html"
    )


# ========================================
# CHECK PAGE
# ========================================

@app.route("/check")
def check():

    if not session.get("logged_in"):

        return redirect("/login")

    return send_file(
        "check.html"
    )


# ========================================
# PREDICT
# ========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # Check login
    if not session.get("logged_in"):

        return jsonify({
            "result": "LOGIN REQUIRED",
            "confidence": "0%"
        }), 401


    # Get JSON data
    data_received = request.get_json()


    # Check received data
    if not data_received:

        return jsonify({
            "result": "NO RESULT",
            "reason": "No news was entered.",
            "confidence": "0%"
        })


    # Get news
    news_text = data_received.get(
        "news",
        ""
    ).strip()


    # Empty input
    if not news_text:

        return jsonify({
            "result": "NO RESULT",
            "reason": "Please enter a news article.",
            "confidence": "0%"
        })


    # ========================================
    # PREPROCESS INPUT
    # ========================================

    clean_input = preprocess_text(
        news_text
    )


    # Check cleaned input
    if not clean_input:

        return jsonify({
            "result": "NO RESULT",
            "reason": "Could not process the entered news.",
            "confidence": "0%"
        })


    # ========================================
    # TF-IDF TRANSFORMATION
    # ========================================

    event_features = vectorizer.transform(
        [clean_input]
    )


    # ========================================
    # PREDICTION
    # ========================================

    prediction = model.predict(
        event_features
    )[0]


    # ========================================
    # CONFIDENCE
    # ========================================

    probabilities = model.predict_proba(
        event_features
    )[0]


    confidence = max(
        probabilities
    ) * 100


    # ========================================
    # RESULT
    # ========================================

    # Since threshold is 0,
    # model prediction will be displayed.
    if confidence < LOW_CONFIDENCE_THRESHOLD:

        display_result = "UNCERTAIN"

    else:

        display_result = prediction


    # ========================================
    # SAVE HISTORY
    # ========================================

    history_item = {

        "text": news_text,

        "result": display_result,

        "confidence": round(
            confidence,
            2
        ),

        "date": datetime.now().strftime(
            "%d-%m-%Y %I:%M %p"
        )
    }


    history_data.append(
        history_item
    )


    save_history()


    # ========================================
    # SEND RESULT
    # ========================================

    return jsonify({

        "result": display_result,

        "confidence": f"{confidence:.2f}%"

    })


# ========================================
# HISTORY PAGE
# ========================================

@app.route("/history")
def history():

    if not session.get("logged_in"):

        return redirect("/login")

    return send_file(
        "history.html"
    )


# ========================================
# HISTORY DATA
# ========================================

@app.route("/history-data")
def history_data_route():

    return jsonify(
        history_data
    )


# ========================================
# CLEAR HISTORY
# ========================================

@app.route(
    "/clear-history",
    methods=["POST"]
)
def clear_history():

    global history_data

    history_data = []

    save_history()

    return jsonify({
        "message": "History cleared successfully"
    })


# ========================================
# REPORT PAGE
# ========================================

@app.route("/report")
def report():

    if not session.get("logged_in"):

        return redirect("/login")

    return send_file(
        "report.html"
    )


# ========================================
# GENERATE PDF REPORT
# ========================================

@app.route("/generate-report")
def generate_report():

    if not session.get("logged_in"):

        return redirect("/login")


    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas


    file_name = "History_Event_Analysis_Report.pdf"


    pdf = canvas.Canvas(
        file_name,
        pagesize=A4
    )


    width, height = A4


    # ========================================
    # TITLE
    # ========================================

    pdf.setFont(
        "Helvetica-Bold",
        18
    )

    pdf.drawString(
        50,
        height - 50,
        "Historical Events Classification and Authenticity Analysis"
    )


    pdf.setFont(
        "Helvetica",
        11
    )


    y = height - 90


    # ========================================
    # MODEL DETAILS
    # ========================================

    pdf.drawString(
        50,
        y,
        f"Model Accuracy: {accuracy * 100:.2f}%"
    )

    y -= 20

    pdf.drawString(
        50,
        y,
        f"Precision: {precision * 100:.2f}%"
    )

    y -= 20

    pdf.drawString(
        50,
        y,
        f"Recall: {recall * 100:.2f}%"
    )

    y -= 20

    pdf.drawString(
        50,
        y,
        f"F1 Score: {f1 * 100:.2f}%"
    )

    y -= 40


    # ========================================
    # HISTORY
    # ========================================

    pdf.setFont(
        "Helvetica-Bold",
        13
    )

    pdf.drawString(
        50,
        y,
        "Prediction History"
    )

    y -= 25


    pdf.setFont(
        "Helvetica",
        10
    )


    for item in history_data:

        text = item["text"]

        result = item["result"]

        confidence = item["confidence"]

        date = item.get(
            "date",
            "Date not available"
        )


        pdf.drawString(
            50,
            y,
            f"Result: {result} | Confidence: {confidence}%"
        )

        y -= 18


        pdf.drawString(
            50,
            y,
            f"Date: {date}"
        )

        y -= 18


        words = text.split()

        line = ""


        for word in words:

            if len(line) + len(word) < 90:

                line += word + " "

            else:

                pdf.drawString(
                    60,
                    y,
                    line
                )

                y -= 15

                line = word + " "


        if line:

            pdf.drawString(
                60,
                y,
                line
            )

            y -= 20


        y -= 10


        if y < 60:

            pdf.showPage()

            y = height - 50

            pdf.setFont(
                "Helvetica",
                10
            )


    pdf.save()


    return send_file(
        file_name,
        as_attachment=True
    )


# ========================================
# LOGOUT
# ========================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")


# ========================================
# START SERVER
# ========================================

if __name__ == "__main__":

    print("================================")
    print("FAKE NEWS CLASSIFIER")
    print("SERVER STARTING...")
    print("================================")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )