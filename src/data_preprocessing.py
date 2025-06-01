import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import category_encoders as ce

class DataPreprocessing:
  
  def __init__(self, df):
    self.df = df

  def null_engineering(self):
    # EverMarried
    df_not_null = self.df[self.df['Ever_Married'].notnull()]
    X_train = df_not_null[['Spending_Score', 'Age']].copy()
    X_train['Spending_Score'] = X_train['Spending_Score'].map({'Low': 0, 'Average': 1, 'High': 2})
    y_train = df_not_null['Ever_Married'].copy()
    df_null = self.df[self.df['Ever_Married'].isnull()]
    X_null = df_null[['Spending_Score', 'Age']].copy()
    X_null['Spending_Score'] = X_null['Spending_Score'].map({'Low': 0, 'Average': 1, 'High': 2})
    minmax = MinMaxScaler()
    X_train = minmax.fit_transform(X_train)
    X_null = minmax.transform(X_null)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_null_pred = lr.predict(X_null)
    self.df.loc[self.df['Ever_Married'].isnull(), 'Ever_Married'] = np.where(y_null_pred == 1, 'Yes', 'No')

    # Graduated
    self.df['Graduated'] = self.df['Graduated'].fillna(self.df['Graduated'].mode()[0])

    # Prosession'
    self.df['Profession'] = self.df['Profession'].fillna('Other')

    # Family_Size
    self.df['Family_Size'] = self.df['Family_Size'].fillna(self.df['Family_Size'].median())

    # Work_Experience
    self.df['Work_Experience'] = self.df['Work_Experience'].fillna(self.df['Work_Experience'].median())

    # Var_1
    self.df['Var_1'] = self.df['Var_1'].fillna(self.df['Var_1'].mode()[0])

    return self.df

  def outlier(self, group_col='Segmentation', cols=['Age', 'Work_Experience', 'Family_Size']):
    df = self.df.copy()
    for col in cols:
        for label in df[group_col].unique():
            df_sub = df[df[group_col] == label]
            Q1 = df_sub[col].quantile(0.25)
            Q3 = df_sub[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = max(Q1 - 1.5 * IQR, 1 if col == 'Family_Size' else 0)
            upper = Q3 + 1.5 * IQR
            isoutlier = (df_sub[col] < lower) | (df_sub[col] > upper)
            median_val = df_sub.loc[~isoutlier, col].median()
            df.loc[df_sub[isoutlier].index, col] = round(median_val) if col == 'Family_Size' else median_val
    self.df = df
    return self.df

  def encoding(self, df_test):
    Ord = OrdinalEncoder(categories=[['Male', 'Female'],['No','Yes'], ['No', 'Yes'], ['Cat_1','Cat_2','Cat_3','Cat_4','Cat_5','Cat_6','Cat_7'],['Low', 'Average', 'High']])
    self.df = self.df.astype(str)
    columns = ['Gender', 'Ever_Married', 'Graduated', 'Var_1', 'Spending_Score']
    self.df[columns] = Ord.fit_transform(self.df[columns])
    df_test[columns] = Ord.transform(df_test[columns])
    self.df['Segmentation'] = self.df['Segmentation'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})
    if 'Segmentation' in df_test.columns:
       df_test['Segmentation'] = df_test['Segmentation'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})
    Gen = ce.TargetEncoder()
    self.df[['Profession']] = Gen.fit_transform(self.df[['Profession']], self.df[['Segmentation']])
    df_test[['Profession']] = Gen.transform(df_test[['Profession']])
    return self.df, df_test

  def process(self, df_test):
    self.df = self.null_engineering()
    self.df = self.outlier()
    self.df, df_test = self.encoding(df_test)
    return self.df, df_test