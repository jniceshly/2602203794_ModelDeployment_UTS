import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class PreprocessingHandler : 
    def __init__ (self, data) : 
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.data = data

    def drop_column(self, columns) : 
        self.data.drop(columns=columns, inplace=True)
    
    def encode_column(self, columns) :
        for col in columns :
            self.data[col] = self.encoder.fit_transform(self.data[col])
    
    def standar_scaling(self, target_columns): 
        for col in target_columns :
            reshaped_data_col = self.data[col].values.reshape(-1, 1)
            self.data[col] = self.scaler.fit_transform(reshaped_data_col)
    
    def replace_column_null_mean(self, target_column) :
        target_column_mean = np.mean(self.data[target_column])
        self.data[target_column].fillna(target_column_mean, inplace=True)
    