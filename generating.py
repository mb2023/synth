# Import necessary libraries
from synthmed import SynthMed
from synthcity.plugins.core.dataloader import GenericDataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd

file_path = 'master_spreadsheet.csv'
df = pd.read_csv(file_path)

target_column = "Endo rem"

traindata, testdata = train_test_split(df, test_size=0.2, random_state=42)

# Wrap in DataLoader
X = GenericDataLoader(traindata, target_column=target_column)
X_test = GenericDataLoader(testdata, target_column=target_column)

synmed = SynthMed(
    X=X,
    X_test=X_test,
    model="ctgan",  # or another available model
    parent_folder=Path().resolve().parent / "outputs"
)

# Run the model
synmed.run_model()
synmed.deep_generative_ensemble()
synmed.evaluate_deep_generative_ensemble()
