# Advertisement click prediction

## General info
The main aim of this project is prediction of the advertisement click using the machine learning. Based on the historical data of advertisement clicks (user behaviour, user profile, etc.) I have made a model to predict who is going to click ad on a website in the future. The analysis includes data analysis, data preparation and creation model by different machine learning models such as Logistic Regression, Linear SVC, Decision Tree, Random Forest and AdaBoost.

### Dataset
The dataset comes from Kaggle and can be find [here](https://www.kaggle.com/datasets/arashnic/ctr-in-advertisement). It includes data of advertisement clicks: user behaviour, user profile, products etc.

### Motivation
The internet marketing is an important part of marketing strategies. The companies prefer to advertise their products on websites and social media platforms. The Ad click is important because it allows to determine whether spending their money on digital advertising is worth or not.  It is significant to targeting the right audience because it spending millions to display the advertisements to the audience who is not want to buy ours products can be costly. 

Nowadays, different types of advertisers and search engines rely on modeling to predict ad CTR (i.e. click-through rate) accurately. The higher CTR shows more interest in that specific campaign, while a lower one indicates that this ad may not be as relevant. High CTR is important because shows that more people are clicking to the website. It also helps to get beter ad position for less money on online platformas such as Google etc.

## Project contains:
- Exploratory Data Analysis of Ad click data - **Ad_click_EDA.ipynb**
- Advertisement click prediction with ML algorithms - **Ad_click_prediction.ipynb**
- Python script to train ML models - **ad_click_models.py**
- Python script to generate predictions from trained model - **prediction_model.py**

## Summary
The project includes prediction of the advertisement click using machine learning methods. Based on historical data of the advertisement clicks (user behaviour, user profile, etc.) I have made a model to predict who is going to click ad on a website in the future. I have started with data analysis to better meet them. Then I have cleaned data and prepared them to the modelling (such as feature engineering). Because the target class variable was imbalanced, I have used the SMOTE method to resolve this problem in data. Next I have applied six different classification algorithms like: Logistic Regression, Linear SVC, Decision Tree, Random Forest and AdaBoost. I evaluated models with a few methods to check which model is the best. I used a accuracy score, f1 score and confusion matrix. Finally the best model was AdaBoost classifier with F1 score of 0.89 and accuracy score of 90%. This model has achaived the best result both in F1 score and accuracy score and this is signalling the characteristics of a reasonably good model with comparision to the others. Additionaly I prepared predictions on the test data with the best trained model i.e. AdaBoost.

## Technologies

**The project is created with:**

- Python 3.6
- libraries: pandas, numpy, scikit-learn, seaborn, matplotlib, imbalanced-learn.

**Running the project:**

To run this project use Jupyter Notebook or Google Colab.

You can run the scripts in the terminal:

    prediction_model.py
    ad_click_models.py
     
