def split_df_for_TSF(df, PERIOD, PREDICTION, dir_to_save):
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
    import pandas as pd

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

