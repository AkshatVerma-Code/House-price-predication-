import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import os 
import joblib

MODEL_FILE = "model.pkl"
PIPLINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    # Numerical pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), 
        ("scaler", StandardScaler())
    ])
    
    # Categorical pipeline
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    # Full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])
    
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Load the data set
    housing = pd.read_csv("housing.csv")

    # Create a stratified test set 
    housing['income_cat'] = pd.cut(housing['median_income'], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing.iloc[test_index].drop("income_cat", axis=1).to_csv("input.csv", index=False)
        housing = housing.iloc[train_index].drop("income_cat", axis=1) 

    # Separate the features and labels
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    # Separate the numerical and categorical attributes 
    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.to_list()
    cat_attribs = ["ocean_proximity"]

    # Build the pipeline
    pipeline = build_pipeline(num_attribs, cat_attribs)

    # Transform the data
    housing_prepared = pipeline.fit_transform(housing)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPLINE_FILE) 
    print("model is trained and saved")
else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPLINE_FILE)

    input_data = pd.read_csv("input.csv")
    transformed_input = pipeline.transform(input_data)  
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions

    input_data.to_csv("output.csv", index=False)
    print("model is loaded and predictions are saved to output.csv")
