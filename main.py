import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


def function(dataframe):

    df_cleaned= dataframe.dropna(subset=['house_size'])
    df_cleaned= df_cleaned.dropna(subset=['price'])
    feature_column= 'house_size'
    X= df_cleaned[[feature_column]]
    y = df_cleaned['price']
    print(df_cleaned.size)
    X_train, X_test,y_train, y_test= train_test_split(X,y, test_size=0.5, random_state= 42)
    model = LinearRegression()
    model.fit(X_train,y_train)

    y_pred= model.predict(X_test)
    mse= mean_squared_error(y_test,y_pred)
    print(f'Mean_square_error: {mse}')

    print(f'Coefficients: {model.coef_}')
    print(f'Intercept: {model.intercept_}')

    # sns.scatterplot(x='house_size', y='price', data= df_cleaned)
    # plt.title('Scatter Plot of House Size vs Price')
    # plt.show()


if __name__ == '__main__':

    df= pd.read_csv("realtor-data.zip.csv")
    function(df)


