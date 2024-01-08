from flask import Flask, render_template, jsonify, request
import joblib
import os

app = Flask(__name__)

models_dir = os.path.abspath('models')


stopwords_file = 'stopwords/indonesian.txt'
stopwords_list = []
with open(stopwords_file, 'r') as file:
    stopwords_list = [line.strip() for line in file]
import re
def casefolding(text):
    text = text.lower()
    text = text.strip(" ")
    text = re.sub(r'[?|$|.|!²_:")(-+,]','',text)
    return text

#tokenizing
def nGramToken(text, ngram=2):
    words = [word for word in text.split(" ") if word not in set(stopwords_list)]
    
    # Keep single words
    if len(words) == 1:
        return words

    temp = zip(*[words[i:] for i in range(0, ngram)])
    ans = [' '.join(ngram) for ngram in temp]
    return ans
def stopword_removal(text):
    filters = stopwords_list
    x = []
    data =[]
    def func(x):
        if x in filters:
            return False
        else:
            return True
    fit = filter(func,text)
    for x in fit:
        data.append(x)
    return data

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    do = []
    for w in text:
        dt = stemmer.stem(w)
        do.append(dt)
    d_clean=[]
    d_clean=" ".join(do)
  
    return d_clean
def preprocessing(str):
    str = casefolding(str)
   
    str = nGramToken(str)
    str = stopword_removal(str)
    str = stemming(str)
    
    return str

clf = joblib.load(os.path.join(models_dir, '../models/random_forest.joblib'))
tfidf_vectorizer = joblib.load(os.path.join(
    models_dir, '../models/tfidf.joblib'))


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/singleInputs', methods=['POST'])
def singleInputs():
    request_json = request.form
    text = request_json["text"]
    if text:
        preprocessed = preprocessing(text)
        
        text_list = [preprocessed]
     
        text_tfidf = tfidf_vectorizer.transform(text_list)

        predictions = clf.predict(text_tfidf).tolist()

        print(predictions)
        return jsonify({'results': predictions})
    else:
        return jsonify({'error': 'No Text Detected'})


if __name__ == '__main__':
    app.run(debug=True)

