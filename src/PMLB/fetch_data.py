import sys

from pmlb import classification_dataset_names, regression_dataset_names, fetch_data

try:
    classification_dir = sys.argv[1]
except:    
    classification_dir = "data/PMLB/classification"

try:
    regression_dir = sys.argv[2]
except:    
    regression_dir = "data/PMLB/regression"

print("downloading classification data:")
for i, dataset_name in enumerate(classification_dataset_names):
    print(f"  {i + 1}: {dataset_name}")
    fetch_data(dataset_name, return_X_y=False, local_cache_dir=classification_dir, dropna=True)

print("downloading regression data:")
for i, dataset_name in enumerate(regression_dataset_names):
    print(f"  {i + 1}: {dataset_name}")
    fetch_data(dataset_name, return_X_y=False, local_cache_dir=regression_dir, dropna=True)