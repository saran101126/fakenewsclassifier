from flask import Flask, send_file, request, jsonify, session, redirect
import pandas as pd
import re
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

app = Flask(__name__)

app.secret_key = "history-event-analysis-secret-key"


# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("dataset.csv")

data["text"] = data["text"].astype(str)
data["label"] = data["label"].astype(str).str.upper().str.strip()


# ==========================================
# TEXT PREPROCESSING
# ==========================================

def preprocess_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Keep only alphabets
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


texts = data["text"].apply(preprocess_text)

y = data["label"].map({
    "REAL": 1,
    "FAKE": 0
})

valid_rows = y.notna()

texts = texts[valid_rows]
y = y[valid_rows].astype(int)


# ==========================================
# TF-IDF FEATURE EXTRACTION
# ==========================================

print("================================")
print("TF-IDF FEATURE EXTRACTION")
print("================================")

vectorizer = TfidfVectorizer(
    lowercase=True,
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(texts)

print("FEATURE MATRIX:", X.shape)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# MODEL TRAINING
# ==========================================

print("================================")
print("TRAINING CLASSIFICATION MODEL")
print("================================")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

print("MODEL TRAINING COMPLETED")


# ==========================================
# MODEL EVALUATION
# ==========================================

test_prediction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    test_prediction
)

precision = precision_score(
    y_test,
    test_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_prediction,
    zero_division=0
)


print("================================")
print("MODEL EVALUATION")
print("================================")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")

print("================================")


# ==========================================
# HISTORY
# ==========================================

history_data = []


# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():
    return send_file("index.html")


# ==========================================
# LOGIN PAGE
# ==========================================

@app.route("/login")
def login_page():
    return send_file("login.html")


# ==========================================
# LOGIN
# ==========================================

@app.route("/login", methods=["POST"])
def login():

    login_data = request.get_json() or {}

    username = login_data.get(
        "username",
        ""
    ).strip()

    password = login_data.get(
        "password",
        ""
    ).strip()

    if username and password:

        session["logged_in"] = True
        session["username"] = username

        return jsonify({
            "success": True,
            "message": "Login successful"
        })

    return jsonify({
        "success": False,
        "message": "Please enter username and password"
    })


# ==========================================
# CHECK PAGE
# ==========================================

@app.route("/check")
def check():

    if not session.get("logged_in"):
        return redirect("/login")

    return send_file("check.html")


# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # Check login
    if not session.get("logged_in"):

        return jsonify({
            "result": "LOGIN REQUIRED",
            "reason": "Please login before checking.",
            "confidence": "0%"
        })


    try:

        event_data = request.get_json() or {}

        event = event_data.get(
            "news",
            ""
        ).strip()


        # ----------------------------------
        # EMPTY INPUT
        # ----------------------------------

        if event == "":

            return jsonify({
                "result": "PLEASE ENTER SOME EVENT",
                "reason": "Please enter a history event.",
                "confidence": "0%"
            })


        # ----------------------------------
        # PREPROCESS
        # ----------------------------------

        processed_event = preprocess_text(event)


        if processed_event == "":

            return jsonify({
                "result": "INVALID EVENT",
                "reason": "Please enter meaningful event text.",
                "confidence": "0%"
            })


        # ----------------------------------
        # TF-IDF TRANSFORMATION
        # ----------------------------------

        event_features = vectorizer.transform(
            [processed_event]
        )


        # ----------------------------------
        # CHECK INPUT SIMILARITY
        # ----------------------------------

        similarity_scores = event_features.dot(
            X_train.T
        )

        max_similarity = similarity_scores.max()


        # ==================================
        # NO RESULT THRESHOLD
        # ==================================

        # If the input is very different
        # from the training data,
        # don't force REAL / FAKE prediction.

        NO_RESULT_THRESHOLD = 0.15


        if max_similarity < NO_RESULT_THRESHOLD:

            return jsonify({

                "result": "NO RESULT",

                "reason":
                    "This event is not sufficiently "
                    "similar to the trained dataset. "
                    "Please enter a history event "
                    "related to the available dataset.",

                "confidence": "0%"

            })


        # ----------------------------------
        # MODEL PREDICTION
        # ----------------------------------

        prediction = model.predict(
            event_features
        )[0]


        # ----------------------------------
        # CONFIDENCE
        # ----------------------------------

        probabilities = model.predict_proba(
            event_features
        )[0]

        confidence = max(probabilities) * 100


        # ----------------------------------
        # RESULT
        # ----------------------------------

        result = (
            "REAL"
            if prediction == 1
            else "FAKE"
        )


        reason = (

            "The machine learning model "
            "classified this history event as "

            f"{result} "

            f"with {confidence:.1f}% "
            "model confidence."

        )


        # ----------------------------------
        # SAVE HISTORY
        # ----------------------------------

        history_data.append({

            "event": event,

            "news": event,

            "result": result,

            "confidence":
                f"{confidence:.1f}%",

            "date":
                datetime.now().strftime(
                    "%d-%m-%Y %I:%M %p"
                )

        })


        # ----------------------------------
        # RETURN RESULT
        # ----------------------------------

        return jsonify({

            "result": result,

            "reason": reason,

            "confidence":
                f"{confidence:.1f}%"

        })


    except Exception as e:

        print(
            "PREDICTION ERROR:",
            str(e)
        )

        return jsonify({

            "result": "ERROR",

            "reason":
                "Unable to process the event.",

            "confidence": "0%"

        }), 500


# ==========================================
# HISTORY PAGE
# ==========================================

@app.route("/history")
def history():

    if not session.get("logged_in"):
        return redirect("/login")

    return send_file("history.html")


# ==========================================
# HISTORY DATA
# ==========================================

@app.route("/history-data")
def history_data_route():

    if not session.get("logged_in"):

        return jsonify([])

    return jsonify(history_data)


# ==========================================
# CLEAR HISTORY
# ==========================================

@app.route("/clear-history", methods=["POST"])
def clear_history():

    if not session.get("logged_in"):

        return jsonify({
            "success": False
        })


    history_data.clear()


    return jsonify({

        "success": True,

        "message":
            "History cleared"

    })


# ==========================================
# REPORT PAGE
# ==========================================

@app.route("/report")
def report():

    if not session.get("logged_in"):
        return redirect("/login")

    return send_file("report.html")


# ==========================================
# GENERATE PDF REPORT
# ==========================================

@app.route("/generate-report")
def generate_report():

    if not session.get("logged_in"):

        return jsonify({

            "success": False,

            "message":
                "Login required"

        })


    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas


    filename = (
        "history_event_analysis_report.pdf"
    )


    pdf = canvas.Canvas(
        filename,
        pagesize=A4
    )


    width, height = A4


    y = height - 50


    # ----------------------------------
    # TITLE
    # ----------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        18
    )

    pdf.drawString(
        50,
        y,
        "History Event Analysis Report"
    )


    y -= 40


    # ----------------------------------
    # DATE
    # ----------------------------------

    pdf.setFont(
        "Helvetica",
        11
    )

    pdf.drawString(

        50,
        y,

        "Generated: "
        + datetime.now().strftime(
            "%d-%m-%Y %I:%M %p"
        )

    )


    y -= 35


    # ----------------------------------
    # MODEL EVALUATION
    # ----------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        12
    )

    pdf.drawString(
        50,
        y,
        "Model Evaluation"
    )


    y -= 20


    pdf.setFont(
        "Helvetica",
        10
    )


    pdf.drawString(
        50,
        y,
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    y -= 15


    pdf.drawString(
        50,
        y,
        f"Precision: {precision * 100:.2f}%"
    )

    y -= 15


    pdf.drawString(
        50,
        y,
        f"Recall: {recall * 100:.2f}%"
    )

    y -= 15


    pdf.drawString(
        50,
        y,
        f"F1 Score: {f1 * 100:.2f}%"
    )


    y -= 35


    # ----------------------------------
    # HISTORY TITLE
    # ----------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        12
    )

    pdf.drawString(
        50,
        y,
        "History Event Classification History"
    )


    y -= 25


    # ----------------------------------
    # NO HISTORY
    # ----------------------------------

    if not history_data:

        pdf.setFont(
            "Helvetica",
            10
        )

        pdf.drawString(

            50,
            y,

            "No history events have "
            "been checked yet."

        )


    # ----------------------------------
    # HISTORY ITEMS
    # ----------------------------------

    else:

        for item in history_data:

            if y < 100:

                pdf.showPage()

                y = height - 50


            pdf.setFont(
                "Helvetica-Bold",
                11
            )

            pdf.drawString(

                50,
                y,

                "Result: "
                + item["result"]

            )


            y -= 18


            pdf.setFont(
                "Helvetica",
                10
            )


            pdf.drawString(

                50,
                y,

                "Confidence: "
                + item["confidence"]

            )


            y -= 18


            pdf.drawString(

                50,
                y,

                "Date: "
                + item["date"]

            )


            y -= 20


            event_text = item.get(

                "event",
                item.get(
                    "news",
                    ""
                )

            )


            words = event_text.split()


            line = ""


            for word in words:

                test_line = (
                    line
                    + word
                    + " "
                )


                if len(test_line) > 90:

                    pdf.drawString(
                        50,
                        y,
                        line
                    )

                    y -= 15

                    line = (
                        word
                        + " "
                    )

                else:

                    line = test_line


            if line:

                pdf.drawString(
                    50,
                    y,
                    line
                )

                y -= 15


            y -= 20


    # ----------------------------------
    # SAVE PDF
    # ----------------------------------

    pdf.save()


    return send_file(
        filename,
        as_attachment=True
    )


# ==========================================
# LOGOUT
# ==========================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    import os

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
