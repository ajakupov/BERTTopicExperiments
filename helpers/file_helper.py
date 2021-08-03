import pandas as pd

def read_ott_deceptive():
    ott_path = "./local_datasets/deceptive-opinion.csv"
    return pd.read_csv(ott_path, encoding='utf-8', sep=",", engine="python")


def get_ott_negative():
    ott_dataframe = read_ott_deceptive()
    ott_dataframe_negative = ott_dataframe[ott_dataframe['polarity']=='negative']
    return ott_dataframe['text']