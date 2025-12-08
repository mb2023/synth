import pandas as pd
from pathlib import Path
from synthcity.plugins.core.dataloader import DataLoader
from synthmed1 import SynthMed

# ------------------------------------------------------------------
# 1. LOAD THE DATA
# ------------------------------------------------------------------

train_path = "/home/mb2023/my_hpc_projects/synthcity-docker-project/generated_data/train_fake_data.csv"
test_path = "/home/mb2023/my_hpc_projects/synthcity-docker-project/generated_data/test_fake_data.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df["lung_cancer"] = train_df["lung_cancer"].astype(int)
test_df["lung_cancer"] = test_df["lung_cancer"].astype(int)

# Survival Analysis variables:
TARGET = "lung_cancer"             
TIME_TO_EVENT = "time_to_event_develop"  

# ------------------------------------------------------------------
# 2. CONVERT TO DataLoader 
# ------------------------------------------------------------------

X = DataLoader.from_df(
    train_df,
    target_column=TARGET,
    time_to_event=TIME_TO_EVENT
)

X_test = DataLoader.from_df(
    test_df,
    target_column=TARGET,
    time_to_event=TIME_TO_EVENT
)

# ------------------------------------------------------------------
# 3. RUN SYNTHETIC MODEL TRAINING + EVALUATION
# ------------------------------------------------------------------

synth = SynthMed(
    X=X,
    X_test=X_test,
    model="adsgan",
    epsilon=1.0,         # required for ADSGAN
    repeats=3,
    parent_folder=Path("./outputs/")
)

synth.run_model()
synth.deep_generative_ensemble()
synth.evaluate_deep_generative_ensemble()

print("✓ Done! Synthetic survival data generated and evaluated.")