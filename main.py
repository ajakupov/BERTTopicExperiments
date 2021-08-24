import pandas as pd

from datetime import datetime

from helpers.file_helper import get_ott_negative
from helpers.experiment_helper import generate_experiments

if __name__ == '__main__':
    negative_reviews = get_ott_negative()
    output = pd.DataFrame()
    for experiment in generate_experiments():
        start = datetime.now()
        output = output.append(experiment.get_result())
        output.to_csv("ExperimentResult.csv", index=False)
        print (experiment.to_string())
        end = datetime.now()
        print(end - start)
