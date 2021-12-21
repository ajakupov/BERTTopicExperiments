import pandas as pd

from datetime import datetime

from helpers.file_helper import get_ott_negative
from helpers.experiment_helper import generate_experiments

if __name__ == '__main__':
    output = pd.DataFrame()

    experiments = generate_experiments()

    counter = 0
    for experiment in experiments:
        start = datetime.now()
        print(experiment.to_string())
        output = output.append(experiment.get_result())
        output.to_csv("ExperimentResult.csv", index=False)
        end = datetime.now()
        print(end - start)
        print("{} out of {}, {}%".format(counter, len(experiments), counter/len(experiments)*100))
        counter += 1
