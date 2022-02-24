# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:28:10 2022

Data Preparation for Project 2 of the Udacity Data Scientist Nanodegree

@author: Benedikt Lechner
"""
    

# import libraries
import pandas as pd
import numpy as np
import copy
from sqlalchemy import create_engine
import sys
import sqlite3

class DataPrepare ():
    """
    Class for preparing data for message classification. It will load input and output data from csv files.
    After that it will clean and transform the data and save it to an sql db

    functions:
    ----------
    load_data: load the data from csv
    
    prepare_and_merge: prepares output data and merges input with outputdata
    
    check_for_duplicates: checks for duplicates and potentially drops them
    
    save_to_db: saves data do specified db

    """
    
    def __init__(self, input_path='', output_path=''):
        """
        init, if data is already given, load it.
    
        Arguments:
        ----------
        input_path: path to input csv file
        output_path: path to output csv file
    
        Returns:
        ----------
        None
    
        """
        
        if(input_path != '' and output_path != ''):
            self.load_data(input_path, output_path)
        
    
    
    def load_data(self, input_path, output_path):
        """
        Loads input and output data from csv files and saves them in calss variables
    
        Arguments:
        ----------
        input_path: path to input csv file
        output_path: path to output csv file
    
        Returns:
        ----------
        None
    
        """
        
        # load input and output data from csv
        self.input_data = pd.read_csv(input_path)
        self.output_data = pd.read_csv(output_path)
        
        
    def prepare_and_merge(self, drop_greater_one = True):
        """
        Prepares the data, by transforming the output into seperate columns for each output class.
        Then it is joined with the input dataframe based on the common id.
        If drop_greater_one == True, rows with output values greater 1 will be dropped.
        The merged dataframe is safed in the class variable merged_data
    
        Arguments:
        ----------
        drop_greater_one: determines wether output values greater one should be dropped or not (default = True)
    
        Returns:
        ----------
        None
    
        """
        
        # select the first row of the categories dataframe and split it to get all output columns
        cat_columns = self.output_data['categories'][0].split(';')
        
        # get all category columns
        output_colnames = [column.split('-')[0] for column in cat_columns]
        
        output = copy.deepcopy(self.output_data)
        
        # create a numpy array with id and corresponding output values, where each output value is represented by its own column  
        output['categories'] = output['categories'].apply(lambda elems: [elem[-1] for elem in elems.split(';')])
        output_np = np.append(np.expand_dims(np.asarray(output['id']), axis=1), np.asarray(list(output['categories'])), axis=1)
        
        # transform array into numeric pandas dataframe
        output_df = pd.DataFrame(data = output_np, columns = ['id'] + output_colnames )
        output_df = output_df.apply(pd.to_numeric)
        
        # join input and transformed output dataframe together by common id
        self.merged_data = self.input_data.join(output_df.set_index('id'), on='id')
        
        # if requested, drop all rows where any output value is greater 1, due to false data
        if (drop_greater_one):
            self.merged_data = self.merged_data[(self.merged_data[output_colnames] <= 1).all(axis=1)]
            
        
    
    
    def check_for_duplicates(self, drop_duplicates = True):
        """
        Checks for duplicates in the merged dataframe.
        If drop_duplicates == True, duplicates will be dropped
    
        Arguments:
        ----------
        drop_duplicates: determines wether to drop duplicates or not (default = True)
    
        Returns:
        ----------
        num_duplicates_start: num duplicates found before dropping
        num_duplicates_end: num duplicates found after dropping
    
        """
        
        # check number of duplicates
        num_duplicates_start = sum(self.merged_data.duplicated())
        num_duplicates_end = num_duplicates_start
        
        # drop duplicates if requested
        if (drop_duplicates):
            self.merged_data = self.merged_data.drop_duplicates()
            
            # check number of duplicates again
            num_duplicates_end = sum(self.merged_data.duplicated())          
        
        
        return num_duplicates_start, num_duplicates_end
    
    
    def to_dummies(self, column_name, drop_original = True):
        """
        Change column with defined name for dummies. If requested remove original column
    
        Arguments:
        ----------
        column_name: name of columns to exchange for dummies
        drop_original: should original colum be dropped
    
        Returns:
        ----------
        col_names: list of names of newly created columns
    
        """
        
        # get current column names
        old_column_names = self.merged_data.columns
            
        # get dummy variables
        dummies = pd.get_dummies(self.merged_data[column_name])

        # join with orginal df
        self.merged_data = pd.concat([self.merged_data, dummies], join='outer', axis = 1, sort=False)

        # drop original column if requested
        if drop_original:
            self.merged_data.drop(column_name, axis=1)

        # get list of new column names and removed one
        new_column_names = self.merged_data.columns

            
        return list(set(old_column_names) - set(new_column_names))
    
    
    def save_to_db(self, file_name, table_name):
        """
        saves merged daframe to a database.
    
        Arguments:
        ----------
        table_name: name of table, e.g. 'desaster_response_table_no_2'
    
        Returns:
        ----------
        None
    
        """
        
        connection = sqlite3.connect(file_name)
        
        # write to database
        self.merged_data.to_sql(table_name, connection, index=False)
        connection.close()
        
        


def main(input_data, output_data, file_name):
    
    # initialize class with data
    DataPreparator = DataPrepare(input_data, output_data)
    
    # prepare data
    DataPreparator.prepare_and_merge()
    
    # replace genre with dummies
    DataPreparator.to_dummies('genre', False)
    
    #check for duplicates
    old_dup, new_dup = DataPreparator.check_for_duplicates()
    print('There were %i duplicates, now there are %i left' %(old_dup, new_dup))
    
    # save to db
    table_name = 'emergency_table'
    DataPreparator.save_to_db(file_name, table_name)
    print('Data successfully processed and saved to file %s and table name %s' %(file_name, table_name))
    
    
    
if __name__ == '__main__':
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3]
    main(arg1, arg2, arg3)