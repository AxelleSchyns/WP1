from argparse import ArgumentParser
import pandas as pd
import re
import numpy as np
from openpyxl import Workbook

# Allows to write in an excel file all results from either uliege, cam or crc file for all models at once


# Create a new workbook and select the active worksheet
wb = Workbook()
ws = wb.active

# Merge cells
ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=5)   # Slot 1 (A1:E1)
ws["A1"] = "Uliege - all"
ws.cell(row = 2, column = 1, value= "T_indexing")
ws.cell(row = 2, column = 2, value= "T_search")
ws.cell(row = 2, column = 3, value= "Top-1")
ws.cell(row = 2, column = 4, value= "Top-5")
ws.cell(row = 2, column = 5, value= "Maj")

parser = ArgumentParser()

parser.add_argument(
    "--log_file",
    type=str,
    required=True,
    help="Path to the log file",
)

parser.add_argument(
    "--excel_file",
    type=str,
    required=True,
    help="Path to the Excel file",
)

args = parser.parse_args()


log_file = args.log_file

# Read the log file
with open(log_file, "r") as file:
    log_content = file.read()

# Extract all numbers using regular expression
numbers = re.findall(r'\b(?:0(?:\.\d+)?|\d+\.\d+|\d+e[+-]?\d+)\b', log_content) #re.findall(r'\b\d+\.\d+\b', log_content)

print(str(len(numbers)) + " numbers were retrieved")
print(numbers)
mul = int(len(numbers) / 16)

for j in range(16):
    ws.cell(row = 3 + j, column = 1, value= np.round(np.float64(numbers[j * mul +3]), 2))
    ws.cell(row = 3 + j, column = 2, value= np.round(np.float64(numbers[j * mul + 17]), 2))
    ws.cell(row = 3 + j, column = 3, value= np.round(np.float64(numbers[j * mul + 4])*100, 2))
    ws.cell(row = 3 + j, column = 4, value= np.round(np.float64(numbers[j * mul + 5])*100, 2))
    ws.cell(row = 3 + j, column = 5, value= np.round(np.float64(numbers[j * mul + 10])*100, 2))

wb.save(args.excel_file)
