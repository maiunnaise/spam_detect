# Importer les bibliothèques nécessaires
from flask import Flask, render_template, request, jsonify
import pickle

# Initialiser l'application Flask
app = Flask(__name__)

# Charger les modèles pré-entraînés
cv = pickle.load(open("models/models/cv.pkl", 'rb'))
clf = pickle.load(open("models/models/clf.pkl", 'rb'))

@app.route('/', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        email = request.form['email']
        tokenized_email = cv.transform([email])
        prediction = clf.predict(tokenized_email)
        return render_template('classify.html', prediction=prediction)

    return render_template('classify.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    email = data['email']
    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    return jsonify({'email':email,'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)

#lancer l'application (une des trois possibilités)
# flask --app insurance_app run
# python insurance_app.py
# avec
# gunicorn insurance_app:app