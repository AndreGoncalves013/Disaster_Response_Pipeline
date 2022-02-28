import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

import pickle


def load_data(database_filepath):
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_messages', engine) 
    df = df.head(100)

    X = df.message
    Y = df[df.columns[4:]]
    category_names = Y.columns

    return X, Y, category_names


def twitter_text_cleaning(text):
    
  text = re.sub(r'@[A-Za-z0-9]+', 'mentionplaceholder', text)
  text = re.sub(r'@[A-Za-zA-Z0-9]+', 'mentionplaceholder', text)
  text = re.sub(r'@[A-Za-z0-9_]+','mentionplaceholder', text)
  text = re.sub(r'@[A-Za-z]+', 'mentionplaceholder', text)
  text = re.sub(r'@[-)]+', 'mentionplaceholder', text)
  text = re.sub(r'#', 'hashtagplaceholder', text)

  return text

def tokenize(text):
    
    text = twitter_text_cleaning(text)
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+1'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)

    report_list = []

    for i, category in enumerate(category_names):
        
        accuracy = accuracy_score(Y_test[category].values, Y_pred[:, i]) 
        precision = precision_score(Y_test[category].values, Y_pred[:, i], average='weighted') 
        recall = recall_score(Y_test[category].values, Y_pred[:, i], average='weighted') 
        f1 = f1_score(Y_test[category].values, Y_pred[:, i], average='weighted') 
        
        report_dict = {
            'category': category,
            'accuracy': round(accuracy,2),
            'precision': round(precision,2),
            'recall': round(recall,2),
            'f1_score': round(f1,2)
        }
    
        report_list.append(report_dict)
    
    results_df = pd.DataFrame(report_list)
    print(results_df.set_index('category'))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()