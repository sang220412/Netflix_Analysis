
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")

# 1. Load the dataset
df = pd.read_csv("C:/Users/Sanghraj/PyCharmMiscProject/CloudWatch_Traffic_Web_Attack.csv")
print("Data loaded. Shape:", df.shape)

# 2. Basic cleaning
df = df.drop_duplicates()
df['creation_time'] = pd.to_datetime(df['creation_time'])
df['end_time'] = pd.to_datetime(df['end_time'])
df['time'] = pd.to_datetime(df['time'])
df['src_ip_country_code'] = df['src_ip_country_code'].str.upper()

# 3. Feature engineering
df['duration_seconds'] = (df['end_time'] - df['creation_time']).dt.total_seconds()

# Fill missing numerical values if any
df['bytes_in'].fillna(df['bytes_in'].median(), inplace=True)
df['bytes_out'].fillna(df['bytes_out'].median(), inplace=True)

# Handle any remaining nulls
df.dropna(subset=['src_ip', 'dst_ip'], inplace=True)

# 4. Scaling numerical features
scaler = StandardScaler()
df[['scaled_bytes_in', 'scaled_bytes_out', 'scaled_duration_seconds']] = scaler.fit_transform(
    df[['bytes_in', 'bytes_out', 'duration_seconds']]
)

# 5. One-Hot Encoding for categorical feature (e.g. country)
try:
    encoder = OneHotEncoder(sparse_output=False)  # For sklearn >= 1.2
except TypeError:
    encoder = OneHotEncoder(sparse=False)  # For sklearn < 1.2

encoded_countries = encoder.fit_transform(df[['src_ip_country_code']])
encoded_df = pd.DataFrame(encoded_countries, columns=encoder.get_feature_names_out(['src_ip_country_code']))


# 6. Isolation Forest for anomaly detection
features = df[['bytes_in', 'bytes_out', 'duration_seconds']]
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = iso_forest.fit_predict(features)
df['anomaly'] = df['anomaly'].apply(lambda x: 'Suspicious' if x == -1 else 'Normal')

# 7. Save for Tableau
df.to_csv("processed_cybersecurity_data.csv", index=False)
print("Processed data saved to processed_cybersecurity_data.csv")

# 8. Plot: Distribution of Bytes In/Out
plt.figure(figsize=(12, 6))
sns.histplot(df['bytes_in'], bins=50, kde=True, color='blue', label='Bytes In')
sns.histplot(df['bytes_out'], bins=50, kde=True, color='red', label='Bytes Out')
plt.legend()
plt.title('Distribution of Bytes In and Bytes Out')
plt.savefig("bytes_distribution.png")
plt.close()

# 9. Plot: Protocol Count
plt.figure(figsize=(10, 5))
sns.countplot(x='protocol', data=df, palette='viridis')
plt.title('Protocol Count')
plt.xticks(rotation=45)
plt.savefig("protocol_count.png")
plt.close()

# 10. Plot: Anomalies (Bytes In vs Bytes Out)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bytes_in', y='bytes_out', hue='anomaly', data=df, palette={'Normal': 'green', 'Suspicious': 'red'})
plt.title('Anomalies in Bytes In vs Bytes Out')
plt.savefig("anomaly_scatter.png")
plt.close()

# 11. Plot: Detection Types by Country
plt.figure(figsize=(12, 6))
detection_by_country = pd.crosstab(df['src_ip_country_code'], df['anomaly'])
detection_by_country.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Detection Types by Country Code')
plt.xlabel('Country Code')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("detection_by_country.png")
plt.close()

# 12. Plot: Time Series of Web Traffic
df.set_index('creation_time', inplace=True)
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['bytes_in'], label='Bytes In', marker='o')
plt.plot(df.index, df['bytes_out'], label='Bytes Out', marker='x')
plt.title('Web Traffic Over Time')
plt.xlabel('Time')
plt.ylabel('Bytes')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("web_traffic_time_series.png")
plt.close()

print("All plots saved. You can now import 'processed_cybersecurity_data.csv' into Tableau.")