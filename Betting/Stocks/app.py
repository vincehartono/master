import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up styles
sns.set(style="darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Path to the parquet file
parquet_path = r"C:\Users\Vince\master\Betting\Stocks\asml_5min.parquet"

# Load the data
try:
    df = pd.read_parquet(parquet_path)
except FileNotFoundError:
    print("Parquet file not found.")
    exit()

# Display basic info
print("Data Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Plot closing price over time
plt.figure()
df['close'].plot(title='ASML Closing Price Over Time')
plt.ylabel('Price')
plt.xlabel('Time')
plt.tight_layout()
plt.show()

# Plot volume over time
plt.figure()
df['volume'].plot(title='ASML Volume Over Time', color='orange')
plt.ylabel('Volume')
plt.xlabel('Time')
plt.tight_layout()
plt.show()
