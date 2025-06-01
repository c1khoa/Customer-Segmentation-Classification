import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineering:
  def __init__(self, df):
    self.df = df
  def augment_feature(self, df_test):
    # ID
    self.df.drop(columns="ID", axis=1, inplace=True)
    if "ID" in df_test.columns:
        df_test.drop(columns="ID", axis=1, inplace=True)
    # Has_Kid
    self.df['Has_Kid'] = (self.df['Family_Size'].astype(float) > 2) & (self.df['Ever_Married'] == 1)
    df_test['Has_Kid'] = (df_test['Family_Size'].astype(float) > 2) & (df_test['Ever_Married'] == 1)

    self.df['Has_Kid'] = self.df['Has_Kid'].astype(int)
    df_test['Has_Kid'] = df_test['Has_Kid'].astype(int)
    # Age_Teen
    self.df['Age_Teen'] = (self.df['Age'].astype(float) <= 26).astype(int)
    df_test['Age_Teen'] = (df_test['Age'].astype(float) <= 26).astype(int)

    return self.df, df_test
  
  def select_feature(self, df_test):
        self.df['Married_Spending'] = self.df['Ever_Married'] * self.df['Spending_Score']
        self.df.drop(['Ever_Married', 'Spending_Score'], axis=1, inplace=True)
        df_test['Married_Spending'] = df_test['Ever_Married'] * df_test['Spending_Score']
        df_test.drop(['Ever_Married', 'Spending_Score'], axis=1, inplace=True)
        return self.df, df_test
  def scaling(self, df_test):

    minmax = MinMaxScaler()
    scaled_features = minmax.fit_transform(self.df.drop('Segmentation', axis=1))
    scaled_df = pd.DataFrame(scaled_features, columns=self.df.columns.drop('Segmentation'))
    scaled_df['Segmentation'] = self.df['Segmentation'].values
    if "Segmentation" in df_test.columns:
        scaled_features = minmax.transform(df_test.drop('Segmentation', axis=1))
        scaled_df_test = pd.DataFrame(scaled_features, columns=df_test.columns.drop('Segmentation'))
        scaled_df_test['Segmentation'] = df_test['Segmentation'].values
    else: 
        scaled_df_test = minmax.transform(df_test)

    return scaled_df, scaled_df_test

  def feature(self, df_test):
    self.df, df_test = self.augment_feature(df_test)
    self.df, df_test = self.select_feature(df_test)
    self.df, df_test = self.scaling(df_test)
    return self.df, df_test