import pandas as pd
import re
import os
import numpy as np
from argparse import ArgumentParser

# Obtain the results in output.xls for 1 model on Cam or crc dataset
# Should be done either by copy pasting output from terminal by running the test_accuracy command in a log file
# or taking from the log file resulting from the scripts for results *ONE* model 
parser = ArgumentParser()
parser.add_argument(
    "--log_file",
    type=str,
    required=True,
    help="Path to the log file",
)
parser.add_argument(
    "-dataset",
    type=str,
    required=True
)
args = parser.parse_args()
log_file = args.log_file
if args.dataset == "crc":
    excel_file = "results/excels/output_crc.xlsx"
else:
    excel_file = "results/excels/output_cam.xlsx"


# Read the log file
with open(log_file, "r") as file:
    log_content = file.read()

# Extract all model names using regular expression
pattern = r'--\s*([^-].*?)\s*--'

matches = re.findall(pattern, log_content)
filtered_matches = [match for match in matches if match.strip()]

# Extract all numbers using regular expression
numbers = re.findall(r'\b(?:0(?:\.\d+)?|\d+\.\d+|\d+e[+-]?\d+)\b', log_content) #re.findall(r'\b\d+\.\d+\b', log_content)

print(str(len(numbers))+" numbers were retrieved for model "+ filtered_matches[0])

columns = ['name', 't_indexing (s)', 't_tot (s)', 't_model (s)', 't_transfer (s)', 't_search (s)', 'Top-1' 	,'Top-5', 'Maj']

sorted_numbers = np.zeros(len(columns)-1)
for i in range(len(numbers)):
    if i < 3:
        pass
    elif i == 3:
        sorted_numbers[0] = np.round(np.float64(numbers[3]), 2)
    elif i == 7:
        pass
    elif i < 8:
        sorted_numbers[i+1] = np.round(np.float64(numbers[i])*100, 2)
    else:
        sorted_numbers[i-7] = np.round(np.float64(numbers[i]), 2)
    
row_data = filtered_matches + list(sorted_numbers)
# Create a DataFrame from the extracted numbers
df = pd.DataFrame([row_data], columns=columns)

if os.path.exists(excel_file):
    # Load existing data
    existing_df = pd.read_excel(excel_file)
    
    # Append new row
    df = pd.concat([existing_df, df], ignore_index=True)
    
# Save updated DataFrame back to Excel

# Save the DataFrame to an Excel file
df.to_excel(excel_file, index=False)
