from flask import Flask, render_template, jsonify, request
from flaskext.mysql import MySQL
import joblib
import os
import pandas as pd

app = Flask(__name__, static_url_path='', 
            static_folder='static',
            template_folder='templates')

mysql = MySQL(app)
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'sentifix'
mysql.init_app(app)

models_dir = os.path.abspath('models')
stopwords_dir = os.path.abspath('stopwords')


stopwords_file =os.path.join(stopwords_dir, '../stopwords/indonesian.txt')
stopwords_list = []
with open(stopwords_file, 'r') as file:
    stopwords_list = [line.strip() for line in file]
import re


from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# Define your preprocessing functions
def casefolding(text):
    text = text.lower()
    text = text.strip(" ")
    text = re.sub(r'[?|$|.|!²_:")(-+,]', '', text)
    return text

def tokenizingText(text, ngram=2):
    words = [word for word in text.split(" ") if word not in stopwords_list]

    # Keep single words
    if len(words) == 1:
        return words

    temp = zip(*[words[i:] for i in range(0, ngram)])
    ans = [' '.join(ngram) for ngram in temp]
    return ans

def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    do = [stemmer.stem(w) for w in text]
    d_clean = " ".join(do)
    return d_clean

def stopword_removal(text):
    filters = stopwords_list
    words = text.split()
    data = [word for word in words if word not in filters]
    return data

def group_elements(input_list):
    grouped_list = [input_list[i] + " " + input_list[i + 1] if i + 1 < len(input_list) else input_list[i] for i in range(0, len(input_list), 2)]
    return grouped_list


def insert_review_predictions(reviews, predictions,dates,divisions):
    print(divisions)
    cursor = mysql.get_db().cursor()

    for review, prediction,date,division in zip(reviews, predictions,dates,divisions):
        label = 'POSITIVE' if prediction == 1 else ('NEUTRAL' if prediction == 2 else 'NEGATIVE')
        cursor.execute(
            "INSERT INTO review (review, analisis,date,division) VALUES (%s, %s,%s,%s)",
            (review, label, date,division)
        )
    
    mysql.get_db().commit()
    cursor.close()

clf = joblib.load(os.path.join(models_dir, '../models/random_forest.joblib'))
tfidf_vectorizer = joblib.load(os.path.join(
    models_dir, '../models/tfidf.joblib'))


@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/input')
def input():
    return render_template('input.html')

@app.route('/file')
def file():
    return render_template('reviews.html')




@app.route('/singleInputs', methods=['POST'])
def singleInputs():
    request_json = request.form
    text = request_json["text"]
    if text:
       
        text = casefolding(text)
        text = tokenizingText(text)
        text = stemming(text)
        text = stopword_removal(text)
        text = group_elements(text)

        text_list = text
     
        text_tfidf = tfidf_vectorizer.transform(text_list)

         # Use predict_proba to get probability estimates
        probabilities = clf.predict_proba(text_tfidf).tolist()

        # Extract the probability for the positive class (assuming binary classification)
        confidence_score = probabilities[0][1] 

        # Make the prediction
        prediction = clf.predict(text_tfidf).tolist()

       
        return jsonify({'results': {'prediction': prediction, 'confidence_score': confidence_score}})
    else:
        return jsonify({'error': 'No Text Detected'})

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    print(file)
    if file:
      

        data = pd.read_excel(file)
        baseReviews = data['Review']
        data['Review']= data['Review'].apply(casefolding)
        data['Review']= data['Review'].apply(tokenizingText)
        data['Review']= data['Review'].apply(stemming)
        data['Review']= data['Review'].apply(stopword_removal)
        data['Review']= data['Review'].apply(group_elements)
        reviews = data['Review']
        dates = data['Timestamp'].astype(str)  # Convert Timestamp to string
        division = data['Divisi']
     
        text_tfidf = tfidf_vectorizer.transform(reviews.apply(' '.join))

        predictions = clf.predict(text_tfidf).tolist()
        insert_review_predictions(baseReviews, predictions, dates, division)
     
        results = []
        for review, prediction, timestamp, divisi in zip(baseReviews, predictions, dates, division):
            result_dict = {
                'text': review,
                'prediction': prediction,
                'timestamp': timestamp,
                'divisi': divisi
            }
            results.append(result_dict)

        return jsonify({'results': results})
    else:
        return jsonify({'error': 'No file uploaded'})
    
@app.route('/api/get_reviews', methods=['GET'])
def get_reviews():
    try:
        cursor = mysql.get_db().cursor()
        cursor.execute("SELECT * FROM review")
        data = cursor.fetchall()
        cursor.close()

        reviews = []
        for row in data:
            review_dict = {
                'id': row[0],
                'review': row[1],
                'division': row[2],
                'analisis': row[3],
                'date':str(row[4])
            }
            reviews.append(review_dict)

        return jsonify({'reviews': reviews})

    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)

