import pandas as pd

def clean_path_column(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract the file name from the full path by removing the header
    df['path'] = df['path'].str.replace(r'^/home/tinyrl/code/articulate-anything/', '', regex=True)
    
    # Save the modified DataFrame back to the same CSV file
    df.to_csv(csv_file, index=False)
    return df

# Example usage
file_name = 'partnet_mobility_embeddings.csv'
cleaned_df = clean_path_column(file_name)
print("First few rows of cleaned data:")
print(cleaned_df['path'].head())