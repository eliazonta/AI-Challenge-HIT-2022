import pandas as pd

def split_df_for_TSF_old(df, PERIOD, PREDICTION, dir_to_save):
    """
    Split the dataframe df in train and test, accoding to the time division.
    @df: the dataframe, must have a datetime field
    @PERIOD: is the dimension of the train (in hours)
    @PREDICTION: is the dimension of the test (in hours)

    for each datetime starting from the minimum in the dataframe,
    split the dataframe in 
    train = df[current_time, current_time + PERIOD]
    and 
    test = df[current_time + PERIOD, current_time + PERIOD + PREDICTION]

    finally save the train and the test with a label in the selected directory
    """
    
    date_min = df['datetime'].min()

    date_max = df['datetime'].max()
    current_data = date_min
    # while i can split the dataframes

    i = 0
    while current_data + pd.offsets.Hour(PERIOD) + pd.offsets.Hour(PREDICTION) <= date_max:

        # train is the dataframe of 0-24 h
        df_train = df[ (df['datetime'] <= current_data + pd.offsets.Hour(PERIOD)) & (df['datetime'] > current_data) ]
        # pred is the dataframe of 24-24+1 h
        df_pred  = df[ (df['datetime'] <= current_data  + pd.offsets.Hour(PERIOD) + pd.offsets.Hour(PREDICTION)) & (df['datetime'] > current_data + pd.offsets.Hour(PERIOD))  ]

        df_train.to_csv(dir_to_save+"train_split_"+str(i)+".csv")
        df_pred.to_csv(dir_to_save+"pred_split_"+str(i)+".csv")

        i = i + 1
        current_data +=  pd.offsets.Hour(PREDICTION)

def split_df_for_TSF(df, PERIOD, PREDICTION):
    """
    Split the dataframe df in train and test, accoding to the time division.
    @df: the dataframe, must have a datetime field
    @PERIOD: is the dimension of the train (in hours)
    @PREDICTION: is the dimension of the test (in hours)

    for each datetime starting from the minimum in the dataframe,
    split the dataframe in 
    train = df[current_time, current_time + PERIOD]
    and 
    test = df[current_time + PERIOD, current_time + PERIOD + PREDICTION]

    finally return the train and the test and the number of points to be predicted
    """    
    date_min = df['datetime'].min()
    date_max = df['datetime'].max()
    current_data = date_min
    i = 0
    data_train = []
    data_test = []

    while current_data + pd.offsets.Hour(PERIOD) + pd.offsets.Hour(PREDICTION) <= date_max:
        # train is the dataframe of 0-24 h
        train_window = df[ (df['datetime'] <= current_data + pd.offsets.Hour(PERIOD)) & (df['datetime'] > current_data) ]
        # we do not want anomalies in training windows
        if((train_window['validation_code'].values != 1).any()):
            current_data +=  pd.offsets.Hour(PREDICTION)
            continue
        
        header={
        'sensor_code':train_window['label'].values[0],
        'in_datetime':train_window['datetime'].values[0],
        }

        features = train_window['value'].reset_index(drop=True).to_dict()

        # pred is the dataframe of 24-24+1 h
        test_window  = df[ (df['datetime'] <= current_data  + pd.offsets.Hour(PERIOD) + pd.offsets.Hour(PREDICTION)) & (df['datetime'] > current_data + pd.offsets.Hour(PERIOD))  ]
        target =  test_window['value'].reset_index(drop=True).to_dict()
        val_label = test_window['validation_code'].reset_index(drop=True)
        n_labels = len(val_label)
        val_label.index = [f"val_{idx}" for idx in val_label.index] 
        val_label = val_label.to_dict()
        # if either features is empty or target, continue
        data_train.append({**header,**features})
        data_test.append({**header,**target,**val_label})

        i = i + 1
        # to avoid too many data 
        current_data +=  pd.offsets.Hour(PREDICTION)

    df_train = pd.DataFrame(data_train)
    df_test = pd.DataFrame(data_test)

    # remove data with NaN in training
    bad_rows = df_train.isnull().values.any(axis=1)
    df_train = df_train.iloc[~bad_rows]
    df_test = df_test.iloc[~bad_rows]

    bad_rows = df_test.isnull().values.any(axis=1)
    df_train = df_train.iloc[~bad_rows].reset_index(drop=True)
    df_test = df_test.iloc[~bad_rows].reset_index(drop=True)
    # bad_rows = pd.isnull(df_test).any(1).to_numpy().nonzero()[0]
    # df_train = df_train.drop(labels=bad_rows,axis=0).reset_index(drop=True)
    # df_test = df_test.drop(labels=bad_rows,axis=0).reset_index(drop=True)

    return df_train,df_test,n_labels
