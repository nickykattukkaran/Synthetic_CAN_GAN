import pandas as pd

file = r"Impersonation_attack_dataset.csv"

# Load CSV files
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
df = load_data(file)

# Calculate the time interval between consecutive rows 
df['TimeInterval'] = df['Timestamp'].diff().fillna(0)
#df["TimeInterval"] = (df["TimeInterval"] * 1_000_000).astype(int)

# Ensure df is an independent copy after filtering
df = df[df['TimeInterval'].astype(float) <= 0.008].copy()

#Drop the 'Timestamp' column 
df = df.drop(columns=['Timestamp'])

#Convert 'RemoteFrame' column to 0 for '000' and 1 for '100' 
df['RemoteFrame'] = df['RemoteFrame'].replace({100: 1, 000: 0})

# Remove the first character from the 'ID' column values 
df['ID'] = df['ID'].apply(lambda x: x[1:] if len(x) > 1 else x)

# Remove spaces in the Payload column
df["Payload"] = df["Payload"].str.replace(" ", "", regex=False)

df.to_csv('Impersonation_for_gen.csv', index=False) 
print("DataFrame exported successfully to Impersonation_for_gen.csv")

