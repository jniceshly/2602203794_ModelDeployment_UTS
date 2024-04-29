from data_handler import DataHandler
from training_handler import TrainingHandler
from preprocessing_handler import PreprocessingHandler

FILE_PATH = "dataset/data_D.csv"
TARGET_COLUNN = "churn"
COLUMNS_TO_DROP = ['CustomerId', 'id', 'Unnamed: 0', 'Surname']
CATEGORY_COLUMNS = ['Gender', 'Geography']
COLUMN_NORMALIZE = ['CreditScore', 'Balance', 'EstimatedSalary']
COLUMN_NULL_TO_MEAN = ['CreditScore']

data_handler = DataHandler(FILE_PATH)

data_handler.load_data()
print("Data before Preprocessing")
data_handler.view_data()

preprocess_handler = PreprocessingHandler(data_handler.data)

# drop columns
preprocess_handler.drop_column(COLUMNS_TO_DROP)

# encode category
preprocess_handler.encode_column(CATEGORY_COLUMNS)

# set mean
for col in COLUMN_NULL_TO_MEAN :
    preprocess_handler.replace_column_null_mean(col)

# scale
preprocess_handler.standar_scaling(COLUMN_NORMALIZE)

print("Data After Preprocessing")
data_handler.view_data()

data_handler.create_input_output(TARGET_COLUNN)

training_handler = TrainingHandler(data_handler.input_data, data_handler.output_data)

training_handler.split_data(0.2)

training_handler.train_model()

training_handler.evaluate_model()

training_handler.create_report()

training_handler.make_model_pickle()
