import pandas as pd

# Input and output file paths
input_file = r"C:\Users\Vince\master\Betting\Simulation_MLB_Claude\mlb_atbat_data_filtered.parquet"
output_file = r"C:\Users\Vince\master\Betting\Simulation_MLB_Claude\mlb_atbat_data_sample.csv"

# Read the Parquet file
try:
    df = pd.read_parquet(input_file)
    
    # Create a 5% sample (adjust percentage as needed)
    sample_size = int(len(df) * 0.05)  # 5% of data
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # Save sample as CSV
    df_sample.to_csv(output_file, index=False)
    
    # Print confirmation and basic info
    print(f"Sample conversion successful! CSV file saved to: {output_file}")
    print(f"Original records: {df.shape[0]}")
    print(f"Sample records: {df_sample.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    # Print column information
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    
    # Display first 5 rows
    print("\nFirst 5 rows of sample:")
    print(df_sample.head(5))
    
except Exception as e:
    print(f"Error converting file: {e}")