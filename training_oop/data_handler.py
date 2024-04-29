import pandas as pd
from sklearn.model_selection import train_test_split

class DataHandler :
    def __init__ (self, file_path): 
        self.file_path = file_path
        self.data = None
        self.input_data = None
        self.output_data = None
    
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        
    def create_input_output(self, target_column):
        if getattr(self, 'data', None) is None :
            print("You must first laod the data\nCall the function load_data")

        self.input_data = self.data.drop(columns=[target_column])
        self.output_data = self.data[target_column]

    def view_data(self) :
        if getattr(self, 'data', None) is None :
            print("You must first laod the data\nCall the function load_data")
            return

        print(self.data)

    def view_input_data(self) : 
        if getattr(self, 'input_data', None) is None:
            print("You haven't created the input and output data")
        print(self.input_data)
    
    def view_output_data(self) :
        if getattr(self, 'output_data', None) is None :
            print("You haven't created the input and output data")

        print(self.output_data)
