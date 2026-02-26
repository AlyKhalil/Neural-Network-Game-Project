import pandas as pd
import numpy as np

def show_data(df):
    print()
    print(df.info())
    print()
    print(df.describe())
    print()
    print(df.isnull().sum())
    print()

# TODO
def read_file(file_path):
    return pd.read_csv(file_path)

#TODO
def shuffle(df):
    # # frac = 1, takes 100% of the df; 
    # random_state=42, gives the same shuffled order when rerunning the code;
    # .reset_index(drop=True), prevents adding the old index as a new column
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

#TODO
def split_data(df, train_portion, validation_portion) -> tuple:
    return (
        df.iloc[:train_portion], 
        df.iloc[train_portion:train_portion+validation_portion], 
        df.iloc[train_portion+val_portion:]
    )

def normalize(mn, mx, st):
    return (st - mn) / (mx - mn)

#TODO
def denormalize():
    pass

if __name__ == "__main__":
    #---Load Data from csv file---
    df = read_file("ce889_dataCollection.csv")
    
    #---Shuffle and Split the data---
    df = shuffle(df=df) 
    
    df_length = len(df)
    train_portion = int(0.7 * df_length)
    val_portion = int(0.15 * df_length)
    test_portion = df_length - train_portion - val_portion

    splt = split_data(df, train_portion, val_portion)
    
    train_df = splt[0]
    val_df = splt[1]
    test_df = splt[2]

    #---Normalize data---
    # Normalisation should be done after splitting to avoid leaking any info of val or test sets through min and max
    # Normalisation and Denormalisation should be done based on the training set
    mn = np.min(train_df)
    mx = np.max(train_df)
    normalized_train_df = normalize(mn, mx, train_df)
    normalized_val_df = normalize(mn, mx, val_df)
    normalized_test_df = normalize(mn, mx, test_df)
    
    normalized_train_df.to_csv('normalized_training_data_ce889_dataCollection.csv', index=False)
    normalized_val_df.to_csv('normalized_validation_data_ce889_dataCollection.csv', index=False)
    normalized_test_df.to_csv('normalized_test_data_ce889_dataCollection.csv', index=False)
