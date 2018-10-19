"""
Created on Mon Jun 11 21:12:10 2018
@author: cthomasbrittain
"""

import sys
import json
#
filename = sys.argv[1]
filepath = sys.argv[2]
pathToWriteProcessedFile = sys.argv[3]

request = sys.argv[4]
request = json.loads(request)

try:
    cols_to_remove = request['columnsToRemove']
    unreasonable_increase = request['unreasonableIncreaseThreshold']
except:
    # If columns aren't contained or no columns, exit nicely
    result = {'status': 400, 'message': 'Expected script parameters not found.'}
    print(str(json.dumps(result)))
    quit()

pathToData = filepath + filename


# Clean Data --------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Importing data transformation libraries
import pandas as pd

# The following method will do the following:a
#   1. Add a prefix to columns based upon datatypes (cat and con)
#   2. Convert all continuous variables to numeric (float64)
#   3. Convert all categorical variables to objects
#   4. Rename all columns with prefixes, convert to lower-case, and replace
#      spaces with underscores.
#   5. Continuous blanks are replaced with 0 and categorical 'not collected'
# This method will also detect manually assigned prefixes and adjust the 
# columns and data appropriately.  
# Prefix key:
# a) con = continuous
# b) cat = categorical
# c) rem = removal (discards entire column)

def add_datatype_prefix(df, date_to_cont = True):    
    import pandas as pd
    # Get a list of current column names.
    column_names = list(df.columns.values)
    # Encode each column based with a three letter prefix based upon assigned datatype.
    # 1. con = continuous
    # 2. cat = categorical
    
    for name in column_names:
        if df[name].dtype == 'object':
            try:
                df[name] = pd.to_datetime(df[name])
                if(date_to_cont):
                    new_col_names = "con_" + name.lower().replace(" ", "_").replace("/", "_")
                    df = df.rename(columns={name: new_col_names})
                else:
                    new_col_names = "date_" + name.lower().replace(" ", "_").replace("/", "_")
                    df = df.rename(columns={name: new_col_names})                    
            except ValueError:
                pass
    
    column_names = list(df.columns.values)
    
    for name in column_names:
        if name[0:3] == "rem" or "con" or "cat" or "date":
            pass
        if df[name].dtype == 'object':
            new_col_names = "cat_" + name.lower().replace(" ", "_").replace("/", "_")
            df = df.rename(columns={name: new_col_names})
        elif df[name].dtype == 'float64' or df[name].dtype == 'int64' or df[name].dtype == 'datetime64[ns]':
            new_col_names = "con_" + name.lower().replace(" ", "_").replace("/", "_")
            df = df.rename(columns={name: new_col_names})
    column_names = list(df.columns.values)
    
    # Get lists of coolumns for conversion
    con_column_names = []
    cat_column_names = []
    rem_column_names = []
    date_column_names = []
    
    for name in column_names:
        if name[0:3] == "cat":
            cat_column_names.append(name)
        elif name[0:3] == "con":
            con_column_names.append(name)
        elif name[0:3] == "rem":
            rem_column_names.append(name)
        elif name[0:4] == "date":
            date_column_names.append(name)
            
    # Make sure continuous variables are correct datatype. (Otherwise, they'll be dummied).
    for name in con_column_names:
        df[name] = pd.to_numeric(df[name], errors='coerce')
        df[name] = df[name].fillna(value=0)
    
    for name in cat_column_names:
        df[name] = df[name].apply(str)
        df[name] = df[name].fillna(value='not_collected')
    
    # Remove unwanted columns    
    df = df.drop(columns=rem_column_names, axis=1)
    return df

# ------------------------------------------------------
# Encoding Categorical variables
# ------------------------------------------------------

# The method below creates dummy variables from columns with
# the prefix "cat".  There is the argument to drop the first column
# to avoid the Dummy Variable Trap.
def dummy_categorical(df, drop_first = True):
    # Get categorical data columns.
    columns = list(df.columns.values)
    columnsToEncode = columns.copy() 

    for name in columns:
        if name[0:3] != 'cat':          
            columnsToEncode.remove(name)

    # if there are no columns to encode, return unmutated.
    if not columnsToEncode:
        return df


    # Encode categories
    for name in columnsToEncode:

        if name[0:3] != 'cat':
            continue

        tmp = pd.get_dummies(df[name], drop_first = drop_first)
        names = {}
        
        # Get a clean column name.
        clean_name = name.replace(" ", "_").replace("/", "_").lower()
        # Get a dictionary for renaming the dummay variables in the scheme of old_col_name + response_string
        if clean_name[0:3] == "cat":
            for tmp_name in tmp:
                tmp_name = str(tmp_name)
                new_tmp_name = tmp_name.replace(" ", "_").replace("/", "_").lower()
                new_tmp_name = clean_name + "_" + new_tmp_name
                names[tmp_name] = new_tmp_name
        
        # Rename the dummy variable dataframe
        tmp = tmp.rename(columns=names)
        
        # join the dummy variable back to original dataframe.
        df = df.join(tmp)
    
    # Drop all old categorical columns
    df = df.drop(columns=columnsToEncode, axis=1)
    return df

# Read the file
df = pd.read_csv(pathToData)

# Drop columns such as unique IDs
try:
    df = df.drop(cols_to_remove, axis=1)
except:
    # If columns aren't contained or no columns, exit nicely
    result = {'status': 404, 'message': 'Problem with columns to remove.'}
    print(str(json.dumps(result)))
    quit()
    
# Get the number of columns before hot encoding
num_cols_before = df.shape[1]

# Encode the data.
df = add_datatype_prefix(df)
df = dummy_categorical(df)

# Get the new dataframe shape.
num_cols_after = df.shape[1]


percentage_increase = num_cols_after / num_cols_before

result = ""

if percentage_increase > unreasonable_increase:
    message = "\"error\": \"Feature increase is greater than unreasonableIncreaseThreshold, most likely a unique id was included."
    result = {'status': 400, 'message': message}
else:
    filename = filename.replace(".csv", "")
    import os
    if not os.path.exists(pathToWriteProcessedFile):
        os.makedirs(pathToWriteProcessedFile)
        
    
    writeFile = pathToWriteProcessedFile + filename + "_encoded.csv"
    df.to_csv(path_or_buf=writeFile, sep=',')
    
    
    # Process the results and return JSON results object
    result = {'status': 200, 'message': 'encoded data', 'path': writeFile}
 
print(str(json.dumps(result)))