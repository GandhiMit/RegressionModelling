from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def function(dataframe, feature_list):
    df_cleaned = dataframe.dropna()

    X = df_cleaned[feature_list]
    y = df_cleaned['price']

    numeric_features = [
        'house_size',
        'zip_code',
        'bath',
        'bed'
    ]
    categorical_features = [
        'city',
        'state'
    ]
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean_square_error: {mse}')

    param_grid = {
        'preprocessor__num__scaler__with_mean': [True, False],
        'preprocessor__num__scaler__with_std': [True, False],
        'preprocessor__cat__onehot__handle_unknown': ['error', 'ignore']
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print("Best hyperparameters: ", grid_search.best_params_)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the test set
    y_pred_best = best_model.predict(X_test)
    mse_best = mean_squared_error(y_test, y_pred_best)
    print(f'Best Mean_square_error: {mse_best}')