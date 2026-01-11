import pandas as pd
import sys

try:
    df = pd.read_csv("data/storage/temp/train_20260110_160349.csv")
    print("Columns:", df.columns.tolist())
    
    if 'label' in df.columns:
        print("\nLabel Distribution:")
        print(df['label'].value_counts(dropna=False))
        
    if 'attack_type' in df.columns:
        print("\nAttack Type Distribution:")
        print(df['attack_type'].value_counts(dropna=False))
        
    # Check for mixed types in label
    if 'label' in df.columns:
        print("\nLabel Dtypes:", df['label'].apply(type).value_counts())

except Exception as e:
    print(e)
