import json
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def print_event_ratio(data, list_group, seq_num, outcome):
    events = data[data["PatientID"].isin(list_group)][outcome].sum()
    ratio = events / len(list_group)
    print(f"Ratio of events in data for sequence {seq_num}:", round(ratio, 2))


def main(outcomes, save_dir):

    '''
    lets create split files for all cases: seq 1 (pre-treatment), 2 (pre-treatment + 2), 
                                               3 (pre-treatment + 3), 4 (pre-treatment + 4)
    
    in test set include patients with all four weeks so we can compare across splits
    make sure that the proportion of events is equal between train and test set
    '''
    
    seq_nums = {1:"two_weeks", 2:"two_weeks", 3:"three_weeks", 4:"four_weeks"}

    #test_data define
    filtered_ds = outcomes[outcomes["four_weeks"]==1] #dataset filtered to include all patients with 4 CT
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    splits = stratified_split.split(filtered_ds["PatientID"].values, filtered_ds["DFS"].values)

    test_index = list(splits)[0][1]
    test_data = list(filtered_ds["PatientID"].values[test_index])


    for i in range(1,5):
        #train data include what is not in test data + cases with correct num of imaging
        
        DB_withIM = outcomes[outcomes[seq_nums[i]]==1]
        patients_withIM = DB_withIM["PatientID"].values

        train_data = [k for k in patients_withIM if k not in test_data]

        print(f"Creating dataset for {seq_nums[i]}. Test ratio: {len(test_data)/(len(test_data)+len(train_data))}")

        # Printing ratio of events in train data
        print("DFS ratio calculation: ")
        print(f"train")
        print_event_ratio(DB_withIM, train_data, seq_nums[i], "DFS")
        print("test")
        print_event_ratio(DB_withIM, test_data, seq_nums[i], "DFS")

        print("OS ratio calculation: ")
        print("train")
        print_event_ratio(DB_withIM, train_data, seq_nums[i], "OS")
        print("test")
        print_event_ratio(DB_withIM, test_data, seq_nums[i], "OS")

        print("-----------------------")

        image_split = {"train" : train_data,
                        "test" : test_data}

        name = save_dir+"data_split_seqNum_"+str(i)+".txt"

        with open(name, 'w') as outfile:
                json.dump(image_split, outfile)
    


if __name__ == "__main__":

    outcome_file = '/scratch/p302386/experimentsLongitudinal/outcomes.xlsx'
    outcome_df = pd.read_excel(outcome_file)
    
    save_dir = '/scratch/p302386/experimentsLongitudinal/data_split/'

    main(outcome_df, save_dir)
