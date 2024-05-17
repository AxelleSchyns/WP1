import pandas as pd
import re
import numpy as np
log_file = "test_output.log"
excel_file = "output.xlsx"

# Read the log file
with open(log_file, "r") as file:
    log_content = file.read()

# Extract all numbers using regular expression
numbers = re.findall(r'\b(?:0(?:\.\d+)?|\d+\.\d+|\d+e[+-]?\d+)\b', log_content) #re.findall(r'\b\d+\.\d+\b', log_content)
#numbers_scientific = re.findall(r'\b\d+\.\d+[eE][-+]?\d+\b', log_content)
print(len(numbers))
#print(len(numbers_scientific))
# insert scientific numbers at the right place: Attention peut etre problématique! -> nombre en e changent de model en model -> a adapter
good_numbers = np.zeros(len(numbers))  
for i in range(len(numbers)):
    """if i < 27:
        good_numbers[i] = numbers[i]
    else:"""
    good_numbers[i] = numbers[i]
#good_numbers[27] = numbers_scientific[0]

with_uncertainty = []
# take the uncertainties and create tuples
for i in range(len(good_numbers)):
    if i > 31 or i == 3:
        if i == 3 or (i > 40 and i < 45) or i > 53:
            with_uncertainty.append(np.round_(good_numbers[i], 2))
        else:
            with_uncertainty.append(np.round_(good_numbers[i]*100, 2))
    elif i < 3 or (i+1)%2 == 0:
        continue
    else:
        if i < 22:
            with_uncertainty.append((np.round_(good_numbers[i]*100, 2), np.round_(good_numbers[i+1]*100, 2)))
        else:
            with_uncertainty.append((np.round_(good_numbers[i]*1000,2), np.round_(good_numbers[i+1]*1000,2)))

columns = ['t_indexing (s)'	,'t_tot (ms)'	,'t_model_tot (ms)'	,'t_model (ms)'	,'t_transfer (ms)'	,'t_search (ms)' 	,'Top-1' 	,'Top-5' 	,'Top-1 proj' 	,'Top-5 proj' 	,'Top-1 sim'	,'Top-5 sim'	,'Maj','Maj proj'	,'Maj sim','	t_tot'	,'t_model (s)'	,'t_transfer (s)'	,'t_search (s)'	,'Top-1' 	,'Top-5' 	,'Top-1 proj'	,'Top-5 proj'	,'Top-1 sim'	,'Top-5 sim'	,'Maj'	,'Maj proj','Maj sim','	t_tot (s)'	,'t_model (s)'	,'t_transfer (s)'	,'t_search (s)' 	,'Top-1'	,'Top-5' 	,'Top-1 proj',	'Top-5 proj',	'Top-1 sim'	,'Top-5 sim'	,'Maj'	,'Maj proj'	,'Maj sim']


# sort numbers given the columns order
sorted_numbers = []
for i in range(len(columns)):
    sorted_numbers.append(0)
for i in range(len(with_uncertainty)):
    if i == 0:
        sorted_numbers[0] = with_uncertainty[i]
    elif i == 1 or i == 2:
        sorted_numbers[i + 5] = with_uncertainty[i]
    elif i == 3 :
        sorted_numbers[12] = with_uncertainty[i]
    elif i == 4 or i == 5:
        sorted_numbers[i + 4] = with_uncertainty[i]
    elif i == 6:
        sorted_numbers[13] = with_uncertainty[i]
    elif i == 7 or i == 8:
        sorted_numbers[i + 3] = with_uncertainty[i]
    elif i == 9:
        sorted_numbers[14] = with_uncertainty[i]
    elif i == 10:
        sorted_numbers[1] = with_uncertainty[i]
    elif i == 11 or i == 12:
        sorted_numbers[i - 8] = with_uncertainty[i]
    elif i == 13:
        sorted_numbers[2] = with_uncertainty[i]
    elif i == 14:
        sorted_numbers[5] = with_uncertainty[i]
    elif i < 24:
        sorted_numbers[i + 4] = with_uncertainty[i]
    elif i < 28:
        sorted_numbers[i - 9] = with_uncertainty[i]
    elif i < 37:
        sorted_numbers[i + 4] = with_uncertainty[i]
    else:
        sorted_numbers[i - 9] = with_uncertainty[i]
    
# Create a DataFrame from the extracted numbers
df = pd.DataFrame([sorted_numbers], columns=columns)

# Save the DataFrame to an Excel file
df.to_excel(excel_file, index=False)
