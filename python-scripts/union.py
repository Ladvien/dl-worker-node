#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 08:30:04 2018

@author: cthomasbrittain
"""
# ------------------------------------------------------
# Paths
# ------------------------------------------------------
try:
    import sys
    import json
    import pandas as pd
except:
    # If there are problems loading the data, quit early.
    result = {'status': 400, 'message': 'Needed Python libraries are not installed.'}
    print(str(json.dumps(result)))
    quit()    

filepath = sys.argv[2]
pathToWriteProcessedFile = sys.argv[3]

request = sys.argv[4]
request = json.loads(request)

filepath_one = filepath + request['dataFileOne']
filepath_two = filepath + request['dataFileTwo']

output_filename = request['outputFilename']
##### TEST ############################
#filepath = '/Users/cthomasbrittain/bit-dl/data/lot-data/lot_encoded/'
#filepath_one = filepath + 'lot_nev_2017_encoded.csv'
#filepath_two = filepath + 'lot_nev_2018_encoded.csv'
#######################################


# Try to load those data.
try:
    df_one = pd.read_csv(filepath_one)
    df_two = pd.read_csv(filepath_two)
except:
    # If there are problems loading the data, quit early.
    result = {'status': 400, 'message': 'Problems loading data files'}
    print(str(json.dumps(result)))
    quit()    

# Union the data
try:
    df_both = df_one.append(df_two)
    # Get rid of annoying indexs
    df_both = df_both[df_both.columns.drop(list(df_both.filter(regex='Unnamed')))]
except:
    # If there are problems loading the data, quit early.
    result = {'status': 400, 'message': 'Problems joining the dataframes.'}
    print(str(json.dumps(result)))
    quit()

# Save the output
try:
    output_filename = pathToWriteProcessedFile + output_filename
    df_both.to_csv(output_filename)
except:
    # Problems writing joined data
    result = {'status': 400, 'message': 'Problems writing joined dataframe.'}
    print(str(json.dumps(result)))
    quit()    

# Send the success message.
result = {'status': 200, 'message': 'Files joined succesfully.', 'filePath': output_filename }
print(str(json.dumps(result)))
