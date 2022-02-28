# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 13:38:58 2022

ML Classifier for Project 2 of the Udacity Data Scientist Nanodegree

@author: Benedikt Lechner
"""
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import sqlite3
import pickle
from word2Vec import KerasWord2VecVectorizer

# download needed nltk packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class Classifier():
    
    def load_data(self, file_name, table_name):
        """
        Loads data from database <dbY with tablename <table_name> as pandas dataframe.
    
        Arguments:
        ----------
        db: Database file load data from
    
        Returns:
        ----------
        df.columns: column names of dataframe
    
        """
        
        # load data from database 
        connection = sqlite3.connect(file_name)
        self.df = pd.read_sql('SELECT * FROM %s'%table_name, connection)
        connection.close()
        
        return self.df.columns

        
    def prepare_data(self, input_columns, output_columns):
        """
        Splits dataframe into input and output.
    
        Arguments:
        ----------
        input_columns: input columns for pipeline
        output_columns: output columns for pipeline
    
        Returns:
        ----------
        None
    
        """
        
        self.X = self.df[input_columns]
        self.Y = self.df[output_columns]
    
    
    def split_data(self, test_part=0.25, random_state = 123):
        """
        Splits data into training and testing part.
    
        Arguments:
        ----------
        test_part: size of testing set (default 0.25 = 25%)
        random_state: random_state for train_test_split for reproduction purposes
    
        Returns:
        ----------
        test_part: test part used
        random_state: random state used
    
        """
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=test_part, random_state=135)

        return test_part, random_state
    
        
    def tokenize(self, replace_urls=True, remove_stopwords = True, use_stemming=False,  lem_pos = []):
        """
        Tokenizes input according to input variables. Different methods are possible.
    
        Arguments:
        ----------
        replace_urls: should urls be replaced with placeholder
        use_stemming: should a stemmer be used
        remove_stopwords: should stopwords be removed
        lem_pos: what positions to lemmatize (n: nouns, v: verbs, a: adjektices, r: adverbs, s: satellite adjectives)
    
        Returns:
        ----------
        None
    
        """
        self.tokens = []
        
        for message in self.X:
            
            # Remove potential urls
            if replace_urls:
                url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                detected_urls = re.findall(url_regex, message)
                for url in detected_urls:
                    message = message.replace(url, "urlplaceholder")
            
            # Normalize case and remove punctuation and double blanks
            message = re.sub(r"[^a-zA-Z0-9]", " ",message.lower()).strip()
            
            # Tokenize text
            tokens = word_tokenize(message)
            
            # stemming
            if use_stemming:
                tokens = [PorterStemmer().stem(t) for t in tokens]
            
            # lemmatize
            if lem_pos:
                wordnetlemmatizer = WordNetLemmatizer()
                for pos in lem_pos:
                    tokens = [wordnetlemmatizer.lemmatize(t, pos=pos) for t in tokens]
            
            # Remove stop words
            if remove_stopwords:
                tokens = [t for t in tokens if t not in stopwords.words('english')]
                
            self.tokens.append(tokens)
            
            
    def _cv_tf_idf_rf(self):
        """
        Sets self.pipeline to CountVectorizer + TfidfTransformer + RandomForestClassifier

        """
        self.pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('rfclf', RandomForestClassifier())
            ])
            
        self.grid_search_parameters = {
            'rfclf__criterion': ['gini', 'entropy']
        }
        
        
        
        
    def _w2v_mlp(self):
        """
        Sets self.pipeline to Word2Vec + Dense NN
        Distinguishes between sole pipeline and Gridsearch

        """

        self.pipeline = Pipeline([
            ('w2v', KerasWord2VecVectorizer(epochs = 2500, batch_size=128, embed_size=270)),
            ('mlpcl', MLPClassifier(early_stopping=True))
            ])
            
        self.grid_search_parameters = {
            'w2v__window_size': [3, 5],
            'mlpcl__hidden_layer_sizes': [(200), (400), (200, 200)],
            'mlpcl__activation': ['relu', 'tanh'],
        }
        

    def train(self, pipeline='cv_tf_idf_rf', grid_search=False):
        
        """
        Start training of model.
        
        Arguments:
        ----------
        pipeline: choose pipeline to select
    
        Returns:
        ----------
        None
        """
        self.gs = grid_search
        
        if pipeline=='cv_tf_idf_rf':
            self._cv_tf_idf_rf()
        
        elif pipeline=='w2v_mlp':
            self._w2v_mlp()

        # switch between eith with or without gridsearch
        if self.gs:
            self.cv = GridSearchCV(self.pipeline, param_grid=self.grid_search_parameters)
            self.cv.fit(self.X_train['message'], self.Y_train)
            
        else: 
            self.pipeline.fit(self.X_train['message'], self.Y_train)

    def test(self):
        
        """
        Evalute model.
        
        Arguments:
        ----------
        metrics: metrics to be evaluated
    
        Returns:
        ----------
        evaluation: evaluation
        """
        # predict values
        Y_pred = []
        if self.gs:
            Y_pred = self.cv.predict(self.X_test['message'])
        else:
            Y_pred = self.pipeline.predict(self.X_test['message'])
        
        # get f1 score per category
        self.f1s = []
        for i, value in  enumerate(self.Y_test.columns):
            f1 = f1_score(list(self.Y_test[value].values), list(Y_pred[:, i]))
            self.f1s.append(f1)

        return self.f1s
    
    
    def export_model(self, filename):
        
        """
        Exports model to pickle file
        
        Arguments:
        ----------
        filename: filename of exported pickle file
    
        Returns:
        ----------
        None
        """
        
        with open(filename, 'wb') as pf:
            if self.gs:
                pickle.dump(self.cv, pf)
            else:
                pickle.dump(self.pipeline, pf)
        with open(filename.split('.')[0] +'_scores.pkl', 'wb') as pf:
            pickle.dump(list(self.f1s), pf)
        
        print('model successfully saved to %s'%filename)
        
        
        
def main(db_file_name, save_path):
    
    # set table name input columns
    table_name = 'emergency_table'
    input_columns = ['message', 'genre']
    pipeline_to_use = 'cv_tf_idf_rf' # options are [cv_tf_idf_rf, w2v_mlp]
    
    # init Classifier, load data and get outpul columns
    Emegency_classifier = Classifier()
    cols = Emegency_classifier.load_data(db_file_name, table_name)
    output_columns = list(set(cols) - set(['id', 'original'] + input_columns))
    
    # prepare and tokenize data
    Emegency_classifier.prepare_data(input_columns, output_columns)
    Emegency_classifier.tokenize(replace_urls=True, remove_stopwords = True, use_stemming=False,  lem_pos = ['n', 'v', 'a', 'r', 's'])
   
    # split data into training and testing
    test_part, random_state = Emegency_classifier.split_data()
    print('Data was split with random state %i and %i%% as testing data'%(random_state, test_part*100))
    
    # run training
    Emegency_classifier.train(pipeline_to_use, True)
    
    # test
    metrics = Emegency_classifier.test()
    print(metrics)
    
    # export model to pickle
    Emegency_classifier.export_model(save_path)
    
    
if __name__ == '__main__':
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    main(arg1, arg2)