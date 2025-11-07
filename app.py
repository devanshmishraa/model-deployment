from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)
cv = pickle.load(open("models/cv.pkl", 'rb'))
clf = pickle.load(open("models/clf.pkl", 'rb'))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/api/predict', methods=['post'])
def predict():
    email = request.form.get('email')
    # predict email
    print(email)
    X = cv.transform([email])
    prediction = clf.predict(X)
    prediction = 1 if prediction == 1 else -1
    return render_template('index.html', response=prediction)


@app.route("/api/predict", methods=["POST"])
def app_predict():
    data = request.get_json(force=True)
    # predict email
    email = data["content"]
    X = cv.transform([email])
    prediction = clf.predict(X)
    prediction = 1 if prediction == 1 else -1
    return jsonify({prediction: prediction})
    


if __name__ == "__main__":
    app.run(debug=True)