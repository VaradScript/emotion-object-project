# analysis/analysis.py (Safe version with debugging)
import pandas as pd
import matplotlib.pyplot as plt
import os

# 📁 Path setup
csv_path = os.path.join("data", "output.csv")
if not os.path.exists(csv_path):
    print("❌ No data found. Please run detection first to generate output.csv.")
    exit()

# ✅ Check for valid header
with open(csv_path, 'r') as f:
    header = f.readline()
    if not header.lower().startswith("timestamp"):
        print("⚠️ Invalid CSV format. First line must be: Timestamp,Detected Object,Emotion")
        exit()

# ✅ Load CSV
df = pd.read_csv(csv_path)

# Debug print
print("📋 Columns:", df.columns.tolist())
print("🔍 Sample data:\n", df.head())

# 🧼 Clean and normalize
if 'Detected Object' not in df.columns or 'Emotion' not in df.columns:
    print("❌ Required columns not found in CSV.")
    exit()

df.dropna(subset=["Detected Object", "Emotion"], inplace=True)
df["Detected Object"] = df["Detected Object"].str.lower().str.strip()
df["Emotion"] = df["Emotion"].str.lower().str.strip()

# Filter only objects of interest
valid_objects = ["cell phone", "book", "other"]
df = df[df["Detected Object"].isin(valid_objects)]

# 🔁 Group for chart
grouped = df.groupby(["Detected Object", "Emotion"]).size().unstack().fillna(0)

if grouped.empty:
    print("❌ No valid grouped data to plot.")
    print("🧪 Check your 'Detected Object' values. Should be only:", valid_objects)
    print("🧾 Existing object values:", df['Detected Object'].unique())
    exit()

# 📊 Plotting
plt.figure(figsize=(12, 6))
grouped.plot(kind="bar", stacked=True, colormap="Set3")
plt.title("📊 Emotion Distribution per Object")
plt.xlabel("Object Detected")
plt.ylabel("Number of Frames")
plt.xticks(rotation=0)
plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# 💾 Save + Show
os.makedirs("analysis", exist_ok=True)
plt.savefig("analysis/emotion_object_chart.png")
print("✅ Chart saved as analysis/emotion_object_chart.png")
plt.show()
