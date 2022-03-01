import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, GridSearchCV

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

import pickle
import gzip


def load_data(database_filepath):
    '''
    INPUT
    database_filepath - file path of the dataset
        
    OUTPUT
    X - Dataframe with model features
    Y - Dataframe with target features
    category_names - list of all target category columns
    '''
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_messages', engine)

    X = df.message
    Y = df[df.columns[4:]]
    category_names = Y.columns

    return X, Y, category_names


def twitter_text_cleaning(text):
    '''
    INPUT
    text - text that might contain mentions and hashtags
        
    OUTPUT
    text - text with mentions and hashtags removed
    '''
    
    text = re.sub(r'@[A-Za-z0-9]+', 'mentionplaceholder', text)
    text = re.sub(r'@[A-Za-zA-Z0-9]+', 'mentionplaceholder', text)
    text = re.sub(r'@[A-Za-z0-9_]+','mentionplaceholder', text)
    text = re.sub(r'@[A-Za-z]+', 'mentionplaceholder', text)
    text = re.sub(r'@[-)]+', 'mentionplaceholder', text)
    text = re.sub(r'#', 'hashtagplaceholder', text)

    return text

def tokenize(text):
    '''
    INPUT
    text - text that will be tokenized
        
    OUTPUT
    clean_tokens - tokens to be used as features to machine learning models
    '''
    
    # Removing mentions and hashtags
    text = twitter_text_cleaning(text)
    text = text.lower()
    text = re.sub("[^\w\s]", " ", text)
    
    # Removing urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+1'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Converting words into tokens
    tokens = word_tokenize(text)

    stop_words = stopwords.words("english")

    tokens = [t for t in tokens if t not in stop_words]

    # Doing lemmatisation process on the tokens
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    '''
    Function that creates the pipeline for data preprocessing and modeling
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        #('clf', MultiOutputClassifier(SVC()))
    ])

    param_grid = {
        'clf__estimator__max_depth': [100, 200],
        'clf__estimator__n_estimators': [5, 10, 20]
    }


    #param_grid = {
    #    'clf__estimator__kernel': ['linear', 'rbf'],
    #    'clf__estimator__C':[1, 10]
    #}

    cv = GridSearchCV(pipeline, param_grid=param_grid, verbose = 4, cv=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Function for calculating classification metrics to evaluate model performance

    INPUT
    model - Trained machine learning model
    X_test = Model features of the test dataset
    Y_test- Target features of the test dataset
    category_names - List with the name of each target feature
    '''

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

    '''
    INPUT
    model - Trained machine learning model
    model_filepath = Path where the model will be saved
    '''

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