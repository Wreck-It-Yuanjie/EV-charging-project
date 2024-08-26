
def preprocess_data(df):
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

# Example usage
# EVdf0 = preprocess_data(EVdf0)
