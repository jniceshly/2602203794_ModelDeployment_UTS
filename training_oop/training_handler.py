import xgboost as xgb 
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

class TrainingHandler :
    def __init__ (self, input_data, output_data) :
        self.xgb = xgb.XGBClassifier(objective="binary:logistic", seed=42)
        self.input_data = input_data
        self.output_data = output_data
        self.input_train = None
        self.output_train = None
        self.input_test = None
        self.output_test = None
        self.prediction = None
    
    def split_data(self, test_percentage) : 
        self.input_train, self.input_test, self.output_train, self.output_test = train_test_split(self.input_data, self.output_data, test_size=test_percentage)

    def train_model(self) :
        self.xgb.fit(self.input_train, self.output_train)
    
    def evaluate_model(self):
        self.prediction = self.xgb.predict(self.input_test)
        print(f"Accuracy Score {accuracy_score(self.output_test, self.prediction)}")
        print(f"F1 Score {f1_score(self.output_test, self.prediction)}")
        print(f"Recall Score {recall_score(self.output_test, self.prediction)}")
        print(f"Precision Score {precision_score(self.output_test, self.prediction)}")
    
    def create_report(self):
        ## Plot importance
        xgb.plot_importance(self.xgb)
        plt.show()

        ## Confusion matrix
        cfm = confusion_matrix(self.output_test, self.prediction)
        
        sns.heatmap(cfm, annot=True, fmt='d')
        plt.show()

    def make_model_pickle(self) : 
        with open('model_pickle/xgboost_model.pkl', 'wb') as f:
            pickle.dump(self.xgb, f)
