import pandas as pd
from utils import utils

if __name__ == '__main__':

    df= pd.read_csv("realtor-data.zip.csv")
    utils.function(df)


