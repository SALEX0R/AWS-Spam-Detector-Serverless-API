from flask import Flask, request
import joblib

app = Flask(__name__)

vectorizer = joblib.load("vectorizer.pkl")
spam_ham_model = joblib.load("spam_ham_model.pkl")

@app.route('/')
def index():
    return "Hello World"

@app.route('/spamorham', methods = ['GET', 'POST'])
#spam or ham "ham " is an email that is not spam
def spamorham():
    message = request.args.get("message")   #http request paramete, retrieve
    vect_message = vectorizer.transform([message])
    result = spam_ham_model.predict(vect_message)[0]
    return result
if __name__ == "__main__":
    app.run()
    #app.run(debug = True)