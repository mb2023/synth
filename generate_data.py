# generate_data.py
import pandas as pd
import random
import os

def generate_fake_data(n: int=100) -> pd.DataFrame:
    """
    Generates a DataFrame with fake patient data.

    Args:
        n (int): The number of records to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the fake data.
    """
    data = []

    for _ in range(n):
        record = {
            "patient_id": _,
            "age": random.randint(18, 99),
            "sex": random.choice(["Male", "Female"]),
            "smoking_status": random.choice(["Never", "Current", "Former"]),
            "physical_activity": random.choice(["Low", "Moderate", "High"]),
            "blood_pressure_systolic": random.randint(80, 200),
            "blood_pressure_diastolic": random.randint(60, 120),
            "bmi": round(random.uniform(16.0, 45.0), 1),
            "diabetes": random.choice([True, False]),
            "hypertension": random.choice([True, False]),
            "heart_disease": random.choice([True, False]),
            "lung_cancer": random.choice([True, False]),
            "time_to_event_develop": random.randint(0, 365 * 10), # Time in days, up to 10 years
        }
        data.append(record)

    return pd.DataFrame(data)

if __name__ == "__main__":
    # Define the number of samples for train and test data
    n_train = 200
    n_test = 100

    print(f"Generating {n_train} training samples...")
    traindata = generate_fake_data(n=n_train)
    print(f"Generating {n_test} testing samples...")
    testdata = generate_fake_data(n=n_test)

    # Define output directory relative to where the script is run
    output_dir = "generated_data"
    os.makedirs(output_dir, exist_ok=True) # Create the directory if it doesn't exist

    # Save the DataFrames to CSV files
    train_output_path = os.path.join(output_dir, "train_fake_data.csv")
    test_output_path = os.path.join(output_dir, "test_fake_data.csv")

    traindata.to_csv(train_output_path, index=False)
    testdata.to_csv(test_output_path, index=False)

    print(f"Successfully generated {n_train} training samples and saved to {train_output_path}")
    print(f"Successfully generated {n_test} testing samples and saved to {test_output_path}")
