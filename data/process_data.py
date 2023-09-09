import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    #merging on left so it will keep all messages 
    df = df_messages.merge(df_categories, how="left", on ="id")
    return df



def clean_data(df):
    #splitting categories
    categories = df["categories"].str.split(pat = ";", expand = True)
    row = categories.iloc[0]
    #renaming categories
    category_colnames = list(map(lambda x: x[:-2], row))
    categories.columns = category_colnames
    #get rid of related-2 values
    categories["related"] = categories["related"].replace('related-2', 'related-1')
    #convert categories values to 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])
    #replace categories column into df
    df = pd.concat([df, categories], axis = 1)
    df = df.drop("categories",axis = 1)

    #remove duplicates
    df = df.drop_duplicates()

    #remove Nan
    df = df.drop("original", axis = 1)

    return df


def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql("disaster_messages", engine, index=False, if_exists='replace') 


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