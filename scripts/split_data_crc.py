import os
import random
import shutil

def split_dataset(source_folder, destination_folder_1, destination_folder_2, split_ratio=0.9):
    # Create destination folders if they don't exist
    os.makedirs(destination_folder_1, exist_ok=True)
    os.makedirs(destination_folder_2, exist_ok=True)

    # Get the list of files in the source folder
    file_list = os.listdir(source_folder)
    for c in file_list:
        os.makedirs(destination_folder_1+"/"+c, exist_ok=True)
        os.makedirs(destination_folder_2+"/"+c, exist_ok = True)
        im_list = os.listdir(source_folder+"/"+c)
        # Calculate the number of files to be moved to the first destination folder
        split_index = int(len(im_list) * split_ratio)

        # Move files to the first destination folder
        for file_name in im_list[:split_index]:
            source_path = os.path.join(source_folder+"/"+c, file_name)
            destination_path = os.path.join(destination_folder_1+"/"+c, file_name)
            shutil.copy(source_path, destination_path)

        # Move remaining files to the second destination folder
        for file_name in im_list[split_index:]:
            source_path = os.path.join(source_folder+"/"+c, file_name)
            destination_path = os.path.join(destination_folder_2+"/"+c, file_name)
            shutil.copy(source_path, destination_path)

def count_per_dataset(index_path, query_path):
    index_list = os.listdir(index_path)
    cpt_test = 0
    for c in index_list:
        print(c, len(os.listdir(index_path+"/"+c)))
        cpt_test += len(os.listdir(index_path+"/"+c))
    print("Index dataset: ", cpt_test)
    
    query_list = os.listdir(query_path)
    cpt_query = 0
    for c in query_list:
        print(c, len(os.listdir(query_path+"/"+c)))
        cpt_query += len(os.listdir(query_path+"/"+c))  
    print("Query dataset: ", cpt_query)
    
if __name__ == "__main__":
    # Replace these paths with your actual paths
    source_folder = '/home/labsig/Downloads/NCT-CRC-HE-100K-NONORM'
    
    destination_folder_1 = '/home/labsig/Documents/Axelle/cytomine/Data/NCT-CRC-HE-100K-NONORM/index' 


    destination_folder_2 = "/home/labsig/Documents/Axelle/cytomine/Data/NCT-CRC-HE-100K-NONORM/query"

    # Specify the split ratio (default is 90-10)
    split_ratio = 0.9

    # Call the function to split the dataset
    #split_dataset(source_folder, destination_folder_1, destination_folder_2, split_ratio)
    count_per_dataset(destination_folder_1, destination_folder_2)
