import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import warnings
import joblib

warnings.simplefilter('ignore')

URL = r'C:\Users\rohit\OneDrive\Desktop\ClickAd\Ad_click_prediction_train (1).csv'

# Define product categories
category_to_interest = {
    1: 'Food',
    2: 'Books',
    3: 'Fashion',
    4: 'Sports',
    5: 'Electronics'
}

# Initialize encoders
product_encoder = LabelEncoder()
interest_encoder = LabelEncoder()
ad_encoder = LabelEncoder()

def clean_data(df):
    df = df.dropna(subset=['gender', 'age_level', 'user_group_id', 'user_depth'])
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    return df

def data_transformation(data):
    df = clean_data(data)

    df['city_development_index'] = df['city_development_index'].fillna(df['city_development_index'].mean()).astype(float)
    df['product_category_2'] = df['product_category_2'].fillna(0).astype(float)
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

    df['product'] = product_encoder.fit_transform(df['product'].astype(str))
    df['user_interest'] = df['product_category_1'].map(category_to_interest).fillna('Unknown')

    np.random.seed(42)
    aligned = np.random.choice([True, False], size=len(df), p=[0.8, 0.2])
    random_ads = np.random.choice(list(category_to_interest.values()), size=len(df))

    df['ad_category'] = np.where(aligned, df['user_interest'], random_ads)

    df['user_interest'] = interest_encoder.fit_transform(df['user_interest'].astype(str))
    df['ad_category'] = ad_encoder.fit_transform(df['ad_category'].astype(str))

    df['interest_match'] = (df['user_interest'] == df['ad_category']).astype(int)

    df = df.drop(['session_id', 'user_id', 'DateTime'], axis=1, errors='ignore')
    return df

def read_data(path):
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None
    return data_transformation(data)

def splitting_data(data):
    X = data.drop('is_click', axis=1).values
    y = data['is_click'].values
    skf = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
    for train_index, test_index in skf.split(X, y):
        return X[train_index], X[test_index], y[train_index], y[test_index]

def f_score(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return round(f1_score(y_test, y_pred, average='weighted'), 3)

def train_models(X_train, X_test, y_train, y_test):
    model_results = []
    best_model = None
    best_score = -1

    classifiers = [
        LogisticRegression(penalty='l2', C=0.01, random_state=0),
        DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=20),
        RandomForestClassifier(max_depth=5, n_estimators=200, criterion='entropy', random_state=0),
        AdaBoostClassifier(n_estimators=200, random_state=0)
    ]

    for classifier in classifiers:
        model = imbpipeline(steps=[
            ('smote', SMOTE()),
            ('scaler', MinMaxScaler()),
            ('classifier', classifier)
        ])

        model.fit(X_train, y_train)
        score = f_score(model, X_test, y_test)
        model_results.append({'Model': classifier.__class__.__name__, 'F1 score': score})

        if classifier.__class__.__name__ == 'AdaBoostClassifier' and score > best_score:
            best_model = model
            best_score = score

    if best_model:
        encoders_bundle = {
            'product': product_encoder,
            'user_interest': interest_encoder,
            'ad_category': ad_encoder,
            'categories': list(category_to_interest.values())  # Save known categories for frontend dropdowns
        }

        joblib.dump(best_model, 'adaboost_ctr_model.pkl')
        joblib.dump(encoders_bundle, 'encoders.pkl')
        print(f"✅ Saved best AdaBoost model with F1 score: {best_score}")

    return pd.DataFrame(model_results).sort_values(by='F1 score', ascending=False).reset_index(drop=True)

if __name__ == '__main__':
    df = read_data(URL)
    if df is None:
        exit("Data loading failed.")
    
    X_train, X_test, y_train, y_test = splitting_data(df)
    print(train_models(X_train, X_test, y_train, y_test))
