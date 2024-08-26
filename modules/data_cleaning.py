
import pandas as pd
import os

def clean_data(df):
    """
    Preprocesses the EV charging data by removing missing values and duplicates.
    
    Parameters:
        df (pd.DataFrame): The raw DataFrame to preprocess.
    
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Drop missing values
    df.dropna(inplace=True)
    
    # Drop duplicate rows
    df.drop_duplicates(inplace=True)
    
    return df

def test_data_cleaning():
    # Path to the dataset in the data folder
    data_file_path = os.path.join('data', 'EVChargingData2010_2020.csv') 

    # Load the dataset
    df = pd.read_csv(data_file_path, low_memory=False)
    
    # Preprocess the data
    cleaned_df = clean_data(df)

    # Save the DataFrame to a CSV file in the /data directory
    cleaned_df.to_csv('./data/cleaned_df.csv', index=False)
    
    # Check if there are no NaN values
    assert not cleaned_df.isnull().values.any(), "There are still missing values in the DataFrame."
    
    # Check if there are no duplicate rows
    assert cleaned_df.duplicated().sum() == 0, "There are still duplicate rows in the DataFrame."
    
    print("All tests passed. No missing values or duplicate rows found. Saved clean data.")


# Test
if __name__ == '__main__':
    test_data_cleaning()