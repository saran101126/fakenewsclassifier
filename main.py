from flask import Flask, send_file, request, jsonify, session, redirect
import pandas as pd
import re
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow import keras
from tensorflow.keras import layers


app = Flask(__name__)
app.secret_key = "fake-news-classifier-secret-key"


# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv("dataset.csv")

data["text"] = data["text"].astype(str)
data["label"] = data["label"].astype(str).str.upper().str.strip()

print("================================")
print("DATASET INFORMATION")
print("================================")

print("LABELS:")
print(data["label"].value_counts())

print("TOTAL NEWS:", len(data))


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


texts = data["text"].apply(preprocess_text)

print("PREPROCESSED SAMPLE:")
print(texts.head(3).tolist())


# =========================================================
# LABEL CONVERSION
# =========================================================

y = data["label"].map({
    "REAL": 1,
    "FAKE": 0
}).values


# =========================================================
# TEXT VECTORIZATION
# =========================================================

max_words = 20000

text_vectorizer = layers.TextVectorization(
    max_tokens=max_words,
    output_mode="tf_idf",
    ngrams=2
)

text_vectorizer.adapt(texts.values)

X = text_vectorizer(texts.values)

print("FEATURE MATRIX SHAPE:", X.shape)


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X.numpy(),
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("TRAINING DATA:", X_train.shape)
print("TESTING DATA :", X_test.shape)


# =========================================================
# DEEP LEARNING MODEL
# =========================================================

model = keras.Sequential([

    layers.Input(
        shape=(X.shape[1],)
    ),

    layers.Dense(
        256,
        activation="relu"
    ),

    layers.Dropout(0.35),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.25),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dropout(0.15),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# =========================================================
# TRAIN MODEL
# =========================================================

print("================================")
print("TRAINING DEEP LEARNING MODEL")
print("================================")

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    verbose=1,
    shuffle=True
)

print("================================")
print("DEEP LEARNING TRAINING COMPLETED")
print("================================")


# =========================================================
# MODEL EVALUATION
# =========================================================

print("================================")
print("MODEL EVALUATION")
print("================================")

test_probability = model.predict(
    X_test,
    verbose=0
).flatten()

test_prediction = (
    test_probability >= 0.5
).astype(int)


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

print("================================")


# =========================================================
# HISTORY
# =========================================================

history_data = []


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return send_file("index.html")


# =========================================================
# LOGIN PAGE
# =========================================================

@app.route("/login")
def login_page():

    return send_file("login.html")


# =========================================================
# LOGIN
# =========================================================

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


    if username == "saran" and password == "143":

        session["logged_in"] = True
        session["username"] = username

        return jsonify({
            "success": True,
            "message": "Login successful"
        })


    return jsonify({
        "success": False,
        "message": "Invalid username or password"
    })


# =========================================================
# CHECK NEWS PAGE
# =========================================================

@app.route("/check")
def check():

    if not session.get("logged_in"):

        return redirect("/login")

    return send_file("check.html")


# =========================================================
# PREDICT NEWS
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    if not session.get("logged_in"):

        return jsonify({
            "result": "LOGIN REQUIRED",
            "reason": "Please login before checking news.",
            "confidence": "0%"
        })


    news_data = request.get_json() or {}

    news = news_data.get(
        "news",
        ""
    ).strip()


    if news == "":

        return jsonify({
            "result": "PLEASE ENTER SOME NEWS",
            "reason": "Please enter a news article.",
            "confidence": "0%"
        })


    # -----------------------------------------------------
    # PREPROCESS NEWS
    # -----------------------------------------------------

    processed_news = preprocess_text(news)


    if processed_news == "":

        return jsonify({
            "result": "INVALID NEWS",
            "reason": "Please enter meaningful news text.",
            "confidence": "0%"
        })


    # -----------------------------------------------------
    # CONVERT NEWS INTO FEATURES
    # -----------------------------------------------------

    news_features = text_vectorizer(
        [processed_news]
    )


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    probability = float(
        model.predict(
            news_features,
            verbose=0
        )[0][0]
    )


    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    if probability >= 0.5:

        result = "REAL NEWS"

        confidence = probability * 100

    else:

        result = "FAKE NEWS"

        confidence = (1 - probability) * 100


    # -----------------------------------------------------
    # REASON
    # -----------------------------------------------------

    reason = (
        "The deep learning model classified "
        f"this news as "
        f"{result.replace(' NEWS', '')} "
        f"with {confidence:.1f}% "
        "model confidence."
    )


    # -----------------------------------------------------
    # SAVE HISTORY
    # -----------------------------------------------------

    history_data.append({

        "news": news,

        "result": result,

        "confidence": f"{confidence:.1f}%",

        "date": datetime.now().strftime(
            "%d-%m-%Y %I:%M %p"
        )
    })


    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return jsonify({

        "result": result,

        "reason": reason,

        "confidence": f"{confidence:.1f}%"
    })


# =========================================================
# HISTORY PAGE
# =========================================================

@app.route("/history")
def history():

    if not session.get("logged_in"):

        return redirect("/login")

    return send_file("history.html")


# =========================================================
# HISTORY DATA
# =========================================================

@app.route("/history-data")
def history_data_route():

    if not session.get("logged_in"):

        return jsonify([])

    return jsonify(history_data)


# =========================================================
# CLEAR HISTORY
# =========================================================

@app.route("/clear-history", methods=["POST"])
def clear_history():

    if not session.get("logged_in"):

        return jsonify({
            "success": False
        })


    history_data.clear()


    return jsonify({
        "success": True,
        "message": "History cleared"
    })


# =========================================================
# REPORT PAGE
# =========================================================

@app.route("/report")
def report():

    if not session.get("logged_in"):

        return redirect("/login")

    return send_file("report.html")


# =========================================================
# GENERATE PDF REPORT
# =========================================================

@app.route("/generate-report")
def generate_report():

    if not session.get("logged_in"):

        return jsonify({
            "success": False,
            "message": "Login required"
        })


    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas


    filename = "fake_news_report.pdf"


    pdf = canvas.Canvas(
        filename,
        pagesize=A4
    )


    width, height = A4

    y = height - 50


    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        18
    )

    pdf.drawString(
        50,
        y,
        "Fake News Classification Report"
    )


    y -= 40


    # -----------------------------------------------------
    # GENERATED DATE
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # MODEL METRICS
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # NEWS HISTORY
    # -----------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        12
    )

    pdf.drawString(
        50,
        y,
        "News Classification History"
    )


    y -= 25


    if not history_data:

        pdf.setFont(
            "Helvetica",
            10
        )

        pdf.drawString(
            50,
            y,
            "No news has been checked yet."
        )


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


            # -------------------------------------------------
            # NEWS TEXT
            # -------------------------------------------------

            news_text = item["news"]

            words = news_text.split()

            line = ""


            for word in words:

                test_line = line + word + " "


                if len(test_line) > 90:

                    pdf.drawString(
                        50,
                        y,
                        line
                    )

                    y -= 15

                    line = word + " "


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


    pdf.save()


    return send_file(
        filename,
        as_attachment=True
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")


# =========================================================
# RUN FLASK SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )