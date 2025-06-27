import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from joblib import load
import warnings

warnings.simplefilter('ignore')

MODELSPATH = r"C:\Users\rohit\OneDrive\Desktop\ClickAd\adaboost_ctr_model.pkl"
URL = r"C:\Users\rohit\OneDrive\Desktop\ClickAd\Ad_Click_prediciton_test.csv"
LABEL_ENCODER = LabelEncoder()


def load_model(model_path):
    '''Load pretrained model'''
    model = load(model_path)
    return model
    

def clean_data(df):
    '''Delete missing data, perform feature engineering for date time feature'''
    df = df.dropna(subset=['gender', 'age_level', 'user_group_id', 'user_depth'])
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    return df


category_to_interest = {
    1: 'Gaming',
    2: 'Books',
    3: 'Fashion',
    4: 'Sports',
    5: 'Electronics'
}

def data_transformation(data):
    '''Fill missing values, convert non-numeric values, and apply feature engineering'''
    df = clean_data(data)
    df['city_development_index'] = df['city_development_index'].fillna('0')
    df['product_category_2'] = df['product_category_2'].fillna('0')
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    
    df['product'] = LABEL_ENCODER.fit_transform(df['product'])

    # Add interest mapping same as training
    df['user_interest'] = df['product_category_1'].map(category_to_interest)

    np.random.seed(42)
    aligned = np.random.choice([True, False], size=len(df), p=[0.8, 0.2])
    random_ads = np.random.choice(list(category_to_interest.values()), size=len(df))

    df['ad_category'] = np.where(aligned, df['user_interest'], random_ads)

    df['user_interest'] = LABEL_ENCODER.fit_transform(df['user_interest'])
    df['ad_category'] = LABEL_ENCODER.fit_transform(df['ad_category'])

    df['interest_match'] = (df['user_interest'] == df['ad_category']).astype(int)

    data = df.drop(['session_id', 'user_id', 'DateTime'], axis=1)
    return data



def read_data(path):
    '''Read data and perform data transformation'''
    data = pd.read_csv(path)
    df = data_transformation(data)
    return df


def get_prediction(test_data):
    '''Generate predictions from test data'''
    test_X = np.array(test_data)
    model = load_model(MODELSPATH)
    predicted = model.predict(test_X)
    test_data['click_prediction'] = predicted
    return test_data


if __name__ == '__main__':
    data = read_data(URL)
    result = get_prediction(data)
    print(result.head())
