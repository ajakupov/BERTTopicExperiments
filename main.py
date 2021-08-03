import itertools
from datetime import datetime

from helpers.file_helper import get_ott_negative
from helpers.experiment_helper import generate_models

if __name__ == '__main__':
    start = datetime.now()
    negative_reviews = get_ott_negative()
    generate_models(negative_reviews)
    end = datetime.now()

    print(end-start)