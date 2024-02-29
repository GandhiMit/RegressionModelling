import pandas as pd
from sklearn.datasets import load_iris
import numpy


def function(dataframe):
    print(dataframe)



if __name__ == '__main__':
    df = pd.read_csv("reviews.csv")
    function(df)
