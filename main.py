import pandas as pd
from utils import utils

if __name__ == '__main__':

    features =['bath','bed', 'zip_code','house_size', 'city', 'state', ]
    df= pd.read_csv("realtor-data.zip.csv")
    utils.function(df, features)


