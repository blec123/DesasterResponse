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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# download needed nltk packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class Classifier():
    
    def load_data(self, db, table_name):
        """
        Loads data from database <dbY with tablename <table_name> as pandas dataframe.
    
        Arguments:
        ----------
        db: Database to load data from
        table_name: table name to load data from
    
        Returns:
        ----------
        df.columns: column names of dataframe
    
        """
        
        # load data from database
        engine = create_engine('sqlite:///EmergencyResponse.db')
        self.df = pd.read_sql('SELECT * FROM desaster_response_table_no_2', engine)
        
        return self.df.columns

        
    def prepare_data(self, input_columns, output_columns, test_part = 0.25, random_state = 123):
        """
        Splits dataframe into input and output. After that splits into training and testing part.
    
        Arguments:
        ----------
        input_columns: input columns for pipeline
        output_columns: output columns for pipeline
        test_part: size of testing set (default 0.25 = 25%)
        random_state: random_state for train_test_split for reproduction purposes
    
        Returns:
        ----------
        test_part: test part used
        random_state: random state used
    
        """
        
        self.X = self.df[input_columns]
        self.Y = self.df[output_columns]
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=test_part, random_state=135)
        
        return test_part, random_state
    
        
    def _cv_tf_idf_rf(self):
        """
        Sets self.pipeline to CountVectorizer + TfidfTransformer + RandomForestClassifier

        """
        self.pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('rfclf', RandomForestClassifier())
            ])
    

    
    
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
    
    
    
    def train(self, pipeline):
        
        """
        Start training of model.
        
        Arguments:
        ----------
        pipeline: choose pipeline to select
    
        Returns:
        ----------
        None
        """
        
        
        
        
    def test(self, metrics):
        
        """
        Evalute model.
        
        Arguments:
        ----------
        metrics: metrics to be evaluated
    
        Returns:
        ----------
        evaluation: evaluation
        """
        
        
        
        
        
        
def main(db, save_path):
    
    # set tablename and input columns
    table_name = 'test_tab'
    input_columns = ['message', 'genre']
    
    # init Classifier, load data and get outpul columns
    Emegency_classifier = Classifier()
    cols = Emegency_classifier.load_data(db, table_name)
    output_columns = list(set(cols) - set(['id', 'original'] + input_columns))
    
    # prepare and tokenize data
    Emegency_classifier.prepare_data(input_columns, output_columns)
    Emegency_classifier.tokenize(replace_urls=True, remove_stopwords = True, use_stemming=False,  lem_pos = ['n', 'v', 'a', 'r', 's'])
    
    # start training
    
    
    
    
    
    
if __name__ == '__main__':
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    main(arg1, arg2)