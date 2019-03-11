import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///%s' %database_filepath)
    df = pd.read_sql_table('mescat_df',con=engine)
    X = df.message.values
    category_names = list(df.columns[3:])
    Y = df[category_names].values
    return X, Y, category_names


def tokenize(text):
    """
    normalize, tokenize and lemmatize the input text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    build a pipeline consisted of several transformers and a final classifier.
    the parameters and the range to tune the model is also defined here.
    the returned object is a classifier
    """
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
                ])

    # use pipeline.get_params().keys() to get the parameter names
    # to avoid that the script takes too long, I used only one parameter in cv.
    parameters = {
            'clf__estimator__n_estimators': [50, 100],
            }

    # instantiate cv
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    model evaluation
    prints out the report of every category on the screen.
    """
    Y_pred = model.predict(X_test)
    for i,col in enumerate(category_names):
        print('\n\n')
        print('######%s######'%col)
        print(classification_report(Y_test[:,i], Y_pred[:,i]))
    return


def save_model(model, model_filepath):
    """
    save the model in the specified path
    """
    with open('%s' %model_filepath, 'wb') as f:
        pickle.dump(model, f)


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
