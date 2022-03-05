import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    '''
    INPUT
    messages_filepath - the file path to the disaster_messages.csv' file
    categories_filepath - the file path to the disaster_categories.csv' file
        
    OUTPUT
    df - Single dataframe with features and target columns for each category
    '''

    # Reading the csv files
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    # Merging messages with categories dataframes
    df = messages_df.merge(categories_df, how='inner', on='id')

    # Creating the target dataframe with each category column
    target_df = df.categories.str.split(";", expand=True)
    category_colnames = target_df.iloc[0,].apply(lambda x:x[:-2])
    target_df.columns = category_colnames

    # Converting category values to numbers 0 or 1
    for column in target_df:
       
        target_df[column] = target_df[column]\
            .apply(lambda x:x[-1])\
            .astype(int).astype(bool)
    
    # Concatenating the original df with the target_df
    df = df.drop(columns='categories', axis=1)
    df = pd.concat([df, target_df], axis=1)

    return df
    



def clean_data(df):
    '''
    INPUT
    df - Pandas Dataframe that might contain duplicates
        
    OUTPUT
    cleaned_df - Dataframe after doing all cleaning process
    '''

    cleaned_df = df.drop_duplicates()
    return cleaned_df


def save_data(df, database_filename):

    '''
    INPUT
    df - Dataframe that will be loaded
    database_filename - Database where the data will be loaded
    '''

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()