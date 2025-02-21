import pandas as pd
import re
import numpy as np
log_file = "results_uliege_one.log"
excel_file = "output_cam.xlsx"

# Read the log file
with open(log_file, "r") as file:
    log_content = file.read()

# Extract all numbers using regular expression
numbers = re.findall(r'\b(?:0(?:\.\d+)?|\d+\.\d+|\d+e[+-]?\d+)\b', log_content) #re.findall(r'\b\d+\.\d+\b', log_content)

print(len(numbers))

columns = ['t_indexing (s)', 't_tot (s)', 't_model (s)', 't_transfer (s)', 't_search (s)', 'Top-1' 	,'Top-5', 'Top-1 proj', 'Top-5 proj', 'Top-1 sim', 'Top-5 sim', 'Maj', 'Maj proj', 'Maj sim', 'F1', 't_tot (s)', 't_model (s)', 't_transfer (s)', 't_search (s)', 'Top-1'     ,'Top-5', 'Top-1 proj', 'Top-5 proj', 'Top-1 sim', 'Top-5 sim', 'Maj', 'Maj proj', 'Maj sim', 'F1']

print(numbers)
sorted_numbers = np.zeros(len(columns))
print(len(columns))
for i in range(len(numbers)):
    if i < 3:
        pass
    elif i == 3:
        sorted_numbers[0] = np.round(np.float64(numbers[3]), 2)
    elif i < 14:
        sorted_numbers[i+1] = np.round(np.float64(numbers[i])*100, 2)
    elif i < 18:
        sorted_numbers[i-13] = np.round(np.float64(numbers[i]), 2)
    elif i < 28:
        sorted_numbers[i+1] = np.round(np.float64(numbers[i])*100, 2)
    else:
        sorted_numbers[i-13] = np.round(np.float64(numbers[i]), 2)
    
# Create a DataFrame from the extracted numbers
df = pd.DataFrame([sorted_numbers], columns=columns)

# Save the DataFrame to an Excel file
df.to_excel(excel_file, index=False)
