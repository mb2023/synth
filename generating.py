# Import necessary libraries
from synthmed import SynthMed
from sklearn.model_selection import train_test_split
from pathlib import Path
import random
import pandas as pd
import numpy as np
import io
from synthcity.plugins.core.dataloader import GenericDataLoader

file_path = 'master_spreadsheet.csv'
df = pd.read_csv(file_path)

target_column = "Endo rem"

traindata, testdata = train_test_split(df, test_size=0.2, random_state=42)
traindata_loader = GenericDataLoader(traindata,target_column=target_column)
testdata_loader = GenericDataLoader(testdata,target_column=target_column)


synmed = SynthMed(
    X=traindata_loader,
    X_test=testdata_loader,
    model="ctgan",
    
    parent_folder=Path().resolve().parent / "outputs"
)

# Run the model
synmed.run_model()

#synthetic_data = synmed.generate_data()
#synthetic_data.to_csv(synmed.parent_folder / synmed.model / "synthetic_data.csv", index=False)
synmed.deep_generative_ensemble()
synmed.evaluate_deep_generative_ensemble()
