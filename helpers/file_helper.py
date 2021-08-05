import pandas as pd

def read_ott_deceptive():
    """
    Either we run it from main or from inside the folder, thus the relative path changes
    """
    try:
        ott_path = "../local_datasets/deceptive-opinion.csv"
        ott_deceptive = pd.read_csv(ott_path, encoding='utf-8', sep=",", engine="python")
    except:
        ott_path = "./local_datasets/deceptive-opinion.csv"
        ott_deceptive = pd.read_csv(ott_path, encoding='utf-8', sep=",", engine="python")
    return ott_deceptive


def get_ott_negative():
    ott_dataframe = read_ott_deceptive()
    ott_dataframe_negative = ott_dataframe[ott_dataframe['polarity']=='negative']
    return ott_dataframe['text']