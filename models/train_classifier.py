import sys
import pandas as pd
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import re

import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM disaster_messages", engine) 
    X = df["message"]
    #takes all rows/observation and all columns/variables 
    # excepts the first 3 (dont take column 0,1,2)
    Y = df.iloc[:,3:]
    category_names = df.columns[3:]
    return X,Y, category_names

def tokenize(text):
    text = re.sub(r'[()\.\;\'\#\!\?\,\--\:\[\]]', '', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(verbose=3)))
    ])
    
    #I dont do the GridSearch bc it take a lot of time and I already did it in ML_Pipeline.ipynb
    #and the best was with a number of trees of 100
    #parameters = {
    #number of trees
    #'clf__estimator__n_estimators' : [50, 100]#,
    #min number of points in a node before node is split
    #'clf__estimator__min_samples_split': [2, 3, 4],
    #min number of points allowed in leaf
    #'clf__estimator__min_samples_leaf': [1, 2, 3, 4]
    #} 
    #cv = GridSearchCV(pipeline, param_grid=parameters)

    #return cv
    return pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for observation, category_names in enumerate(Y_test):
        print(category_names, classification_report(Y_test[category_names], y_pred[:, observation]))

def save_model(model, model_filepath):
    with open('model.pkl', 'wb') as model_filepath:
        pickle.dump(model, model_filepath)


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