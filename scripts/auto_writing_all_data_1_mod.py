from argparse import ArgumentParser
import re
import numpy as np
import os
import pandas as pd

# Allows to write in an excel file results for all three datasets for 1 model (only 'all' for ULiege!)
# (input should be of the type of results.txt in the models' personal folder)
# Writes in a new sheet of an existing excel file or creates a new one if it does not exist
parser = ArgumentParser()
parser.add_argument(
    "--log_file",
    type=str,
    required=True,
    help="Path to the log file",
)
args = parser.parse_args()
log_file = args.log_file
excel_file = "results/excels/results_all_data.xlsx"


# Read the log file
with open(log_file, "r") as file:
    log_content = file.read()

# Extract all model names using regular expression
pattern = r'--\s*([^-].*?)\s*--'

matches = re.findall(pattern, log_content)
filtered_matches = [match for match in matches if match.strip()]
name = filtered_matches[0]

# Extract all numbers using regular expression
numbers = re.findall(r'\b(?:0(?:\.\d+)?|\d+\.\d+|\d+e[+-]?\d+)\b', log_content) #re.findall(r'\b\d+\.\d+\b', log_content)
print(len(numbers))
if len(numbers) > 42:
    numbers = numbers[0:42]
print(str(len(numbers))+" numbers were retrieved for model "+ filtered_matches[0])
print(numbers)

columns = ['name', 'U t_indexing (s)', 'U t_tot (s)', 'U t_model (s)', 'U t_transfer (s)', 'U t_search (s)', 'U Top-1' 	,'U Top-5', 'U Top-1 proj', 'U Top-5 proj', 'U Top-1 sim', 'U Top-5 sim', 'U Maj', 'U Maj proj', 'U Maj sim', 'Crc t_indexing (s)', 'Crc t_tot (s)', 'Crc t_model (s)', 'Crc t_transfer (s)', 'Crc t_search (s)', 'Crc Top-1' 	,'Crc Top-5', 'Crc Maj', 't_indexing (s)', 't_tot (s)', 't_model (s)', 't_transfer (s)', 't_search (s)', 'Top-1' 	,'Top-5', 'Maj']

sorted_numbers = np.zeros(len(columns)-1)
for i in range(len(numbers)):
    # ULiege i = 24 to i=41 ; columns 0 to 13
    # CrC i=0 to i = 11, columns 14 to 21
    # Cam i=12 to i =23, columns 22 to 29
    if i < 3 or i == 7:
        pass
    elif i == 3: # CRC indexing
        sorted_numbers[14] = np.round(np.float64(numbers[3]), 2)
    elif i < 7: # CRC accuracies
        sorted_numbers[i+15] = np.round(np.float64(numbers[i])*100, 2)
    elif i < 12: # CRC times 
        sorted_numbers[i+7] = np.round(np.float64(numbers[i]), 2)
    elif i < 15 or i == 19:
        pass
    elif i == 15: # Cam indexing
        sorted_numbers[22] = np.round(np.float64(numbers[15]), 2)
    elif i < 19: # cam accuracies 
        sorted_numbers[i+11] = np.round(np.float64(numbers[i])*100, 2)
    elif i < 24: # Cam times 
        sorted_numbers[i+3] = np.round(np.float64(numbers[i]), 2)
    elif i < 27 or i == 37:
        pass
    elif i == 27: # Cam indexing
        sorted_numbers[0] = np.round(np.float64(numbers[27]), 2)
    elif i < 37: # cam accuracies 
        sorted_numbers[i-23] = np.round(np.float64(numbers[i])*100, 2)
    else: # Cam times 
        sorted_numbers[i-37] = np.round(np.float64(numbers[i]), 2)
    
row_data = [name] + list(sorted_numbers)
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
