# ------------------------------------------------------
# Load DL Node Configuration
# ------------------------------------------------------
import os
import sys
import errno
import json

try:
    cwd = os.path.dirname(os.path.realpath(__file__))
    worker_node_cfg = ''
    with open(cwd + '/worker-node-configure.json') as f:
        worker_node_cfg = json.load(f)    
    
    try:
        root = worker_node_cfg['root']
        write_path = worker_node_cfg['writePath']
        data_path = worker_node_cfg['dataPath']
    except:
        result = {'status': 400, 'error': 'There was a problem with the configuration file.'}
        print(str(json.dumps(result)))
        quit()
except:
    worker_node_cfg = ''

# Check to see if this is a boss-node request
try:
    request = sys.argv[1]
    request = json.loads(request)
except:
    # Otherwise, try to find the local test request
    test_file_name = 'nn-job-request.json'
    try:
        test_file_path = root + ' /preprocessing-services/models/'    
        with open(test_file_path + test_file_name) as f:
            data = json.load(f)
    except:
        # If both fail, let the user know it didn't work out.
        result = {'status': 200, 
          'error': 'No Job info found and local test not setup.'}
        print(str(json.dumps(result)))
        quit()

# ------------------------------------------------------
# Training Session ID path generation
# ------------------------------------------------------
import string
import random
import datetime

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

id = id_generator()
exe_count = 1
  
# ------------------------------------------------------
# Save Job locally
# ------------------------------------------------------
saved_parameter_file = write_path + 'past_jobs/' + id + '_job.json'
if not os.path.exists(os.path.dirname(saved_parameter_file)):
    try:
        os.makedirs(os.path.dirname(saved_parameter_file))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
with open(saved_parameter_file, 'w') as f:
    f.write(str(json.dumps(request)))

# ------------------------------------------------------
# Load Paths
# ------------------------------------------------------
projectName = request['projectName']
data_file_name = request['dataFileName']
data_file_path = data_path + data_file_name

# ------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------
dep_var = request['depedentVariable']
batchSize = request['batchSize']

epochs = request['epochs']

 # What % of epochs without improvement to lower learning rate
patienceRate = request['patienceRate']

slowLearningRate = request['slowLearningRate']
patience = epochs*patienceRate
loss = request['loss']
pcaComponents = request['pcaComponents']
extra_trees_keep_thresh = request['extraTreesKeepThreshd']
saveWeightsOnlyAtEnd = True
optimizer = request['optimizer']
last_layer_activator = request['lastLayerActivator']

layers = request['hiddenLayers']
learning_rate = request['learningRate']
l1 = request['l1']
l2 = request['l2']
min_dep_var = request['minDependentVarValue']
max_dep_var = request['maxDependentVarValue']
scaler_type = request['scalerType']

cross_val_scoring_type = request['crossValidationCrossingType']
cross_val_only = request['crossValidateOnly']

# ------------------------------------------------------
# Setup Optimizer
# ------------------------------------------------------
import keras

if optimizer == 'adam':
    optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
elif optimizer == 'rmsprop':
    optimizer = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
elif optimizer == 'adadelta':
    optimizer = keras.optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=0.0)    
elif optimizer == 'nadam':
    optimizer = keras.optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
else:
    print('No optimizer')
    quit()

# ------------------------------------------------------
# Load pre-encoded data
# ------------------------------------------------------

import pandas as pd
import numpy as np

df = pd.read_csv(data_file_path)
df = df.fillna(0)
df = df.reindex(np.random.permutation(df.index))

# ------------------------------------------------------
# Remove clients with too long or short dep_var
# ------------------------------------------------------

# It would probably be better to look at the frequency
# of stays, with the assumption a client who's shelter
# stay was not closed out would have a long contiguous 
# stay

df = df.loc[df[dep_var] < max_dep_var]
df = df.loc[df[dep_var] > min_dep_var]

# ------------------------------------------------------
# Place Dependent Variable at end of dataframe
# ------------------------------------------------------

def move_dependent_var_to_end(df, dv_col_name):
    cols = list(df.columns.values) #Make a list of all of the columns in the df
    cols.pop(cols.index(dv_col_name)) #Remove b from list
    df = df[cols+[dv_col_name]] #Create new dataframe with columns in the order you want
    return df

df = move_dependent_var_to_end(df, dep_var)

# -------------------------------------------------------------------------------
# Setup scaler
# -------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler, MinMaxScaler
if scaler_type == 'min_max':
    scale = MinMaxScaler()
elif scaler_type == 'standard':
    scale = StandardScaler()

# -------------------------------------------------------------------------------
# Create Sample X, Y
# -------------------------------------------------------------------------------

df_samp = df.sample(1000)

df_samp = df_samp[df_samp.columns.drop(list(df_samp.filter(regex='nan')))]
X_samp = df_samp.iloc[:, 0:len(df_samp.columns) - 1].values
X_samp = scale.fit_transform(X_samp)
 
y_samp = df_samp.iloc[:,-1].values

# -------------------------------------------------------------------------------
# Feature Extraction with RFE Logistic
# -------------------------------------------------------------------------------
#from pandas import read_csv
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
## feature extraction
#model = LogisticRegression()
#rfe = RFE(model, 50)
#fit = rfe.fit(X_samp, y_samp)
#print("Num Features: " +  str(fit.n_features_))
#print("Selected Features: " + str(fit.support_))
#rank = fit.ranking_
#print("Feature Ranking:" + str(fit.ranking_))
#
#col_names = list(df.columns.values)
#col_names.pop()
#
#cols_ranked = pd.DataFrame({'features': col_names, 'rank': list(rank)})
#
## -------------------------------------------------------------------------------
## Feature Extraction with RFE Linear
## -------------------------------------------------------------------------------
#from pandas import read_csv
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LinearRegression
## feature extraction
#model = LinearRegression()
#
## Apply Recusrive Feature Elementation
#rfe = RFE(model, 50)
#fit = rfe.fit(X, y)
#
## Get an array of the features ranked
#rank = fit.ranking_
#
## Creae a dataframe of the column names by ranking.
#col_names = list(df.columns.values)
#col_names.pop()
#cols_ranked = pd.DataFrame({'features': col_names, 'rank': list(rank)})

# ------------------------------------------------------
# Extremely Randomized Trees
# ------------------------------------------------------
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X_samp, y_samp)

# Get an array of the features ranked
rank = model.feature_importances_

# Creae a dataframe of the column names by ranking.
col_names = list(df_samp.columns.values)
col_names.pop()
cols_ranked = pd.DataFrame({'features': col_names, 'rank': list(rank)})
cols_ranked['rank'] -= cols_ranked['rank'].min()
cols_ranked['rank'] /= cols_ranked['rank'].max()
important_cols = cols_ranked.loc[cols_ranked['rank'] >= extra_trees_keep_thresh]

important_cols = list(important_cols['features'])
important_cols.append(dep_var)

df = df[important_cols]

# ------------------------------------------------------
# Setup ANN
# ------------------------------------------------------
df = df[df.columns.drop(list(df.filter(regex='nan')))]
X = df.iloc[:, 0:len(df.columns) - 1].values
X = scale.fit_transform(X)
y = df.iloc[:,-1].values

# ------------------------------------------------------
# PCA
# ------------------------------------------------------
if pcaComponents > 0:
    from sklearn.decomposition import PCA
    pca = PCA(0.45)
    pca.fit(X)
    X = pca.transform(X)

# ------------------------------------------------------
# Split into Training and Test
# ------------------------------------------------------

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# ------------------------------------------------------
# Network layout
# ------------------------------------------------------
from keras import regularizers
def pileLayers(shapeSize, optimizer, loss, layers):
    model = Sequential()
    model.add(Dense(int(shapeSize*layers[0]['widthModifier']), input_dim=shapeSize, init='normal', activation=layers[0]['activation']))
    for layer in layers[1:]:
        model.add(Dense(int(shapeSize*layer['widthModifier']), 
                        input_dim=shapeSize, init='normal', 
                        kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), 
                        activation=layer['activation']))
        if layer['dropout'] > 0:
            model.add(Dropout(rate = layer['dropout']))
    if last_layer_activator:
        model.add(Dense(1, activation=last_layer_activator))
    else:
        model.add(Dense(1))
    model.compile(loss=loss, optimizer = optimizer, metrics=[loss])
    return model


model = pileLayers(X.shape[1], optimizer, loss, layers)

X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)

# ------------------------------------------------------
# Create save log for postmortem
# ------------------------------------------------------
import os
rootFolder=write_path + "/training-log/"
projectPath=rootFolder + projectName + "_" + id + "_epochs=" + str(epochs)
filepath=projectPath + "/" + projectName + "--{epoch:02d}-{loss:.2f}.hdf5"

if not os.path.exists(projectPath):
    os.makedirs(projectPath)

# ------------------------------------------------------
# NN Callbacks
# ------------------------------------------------------

# Create Keras Checkpoint for logging weights.
if not saveWeightsOnlyAtEnd:
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')


# Create TensorBoard log file, for vizualizing training performance.
tensorBoardLog = TensorBoard(log_dir=write_path + "/training_log" + "/tensorflow_log_" + id, histogram_freq=0, 
                   batch_size=batchSize, 
                   write_graph=True, 
                   write_grads=False,  
                   write_images=False, 
                   embeddings_freq=0, 
                   embeddings_layer_names=None, 
                   embeddings_metadata=None)

class TestCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

testCallback = TestCallback()

# Lower learning rate at plateaus
reduceLROnPlateau = ReduceLROnPlateau(monitor='loss', 
                                      factor=slowLearningRate, 
                                      patience=patience, 
                                      verbose=0, 
                                      mode='auto', 
                                      cooldown=0, 
                                      min_lr=0.0001)
if not saveWeightsOnlyAtEnd:
    callbacks_list = [checkpoint, tensorBoardLog, reduceLROnPlateau, testCallback]
else:
    callbacks_list = [tensorBoardLog, reduceLROnPlateau, testCallback]
# ------------------------------------------------------
# Load saved weights
# ------------------------------------------------------
# model.load_weights('/home/dl/Desktop/length-of-homelessness-nn-trainer/training-log/XC92LG_weights.HDF5')

# ------------------------------------------------------
# Saving training information
# ------------------------------------------------------
import pprint
modelAsJson = pprint.pformat(model.to_json())

trainingInfoFilePath = rootFolder + "training_info" + "_" +  id + "_epochs=" + str(epochs) + ".txt"
f = open(trainingInfoFilePath,"w+")
f.write("Batch Size = " + str(batchSize) + "\n")
f.write("Important Columns: \n")
f.write(str(important_cols) + "\n\n")
f.write("Epochs = " + str(epochs) + "\n")
f.write("patienceRate = " + str(patienceRate) + "\n")
f.write("slowLearningRate = " + str(slowLearningRate) + "\n")
f.write("patience = " + str(patience) + "\n")
f.write("Loss f(x) = " + str(loss) + "\n")
f.write("PCA Components = " + str(pcaComponents) + "\n")
f.write("\n")
f.write("Model Summary:\n")
f.write("\n")
f.write(modelAsJson)
f.close()



# ------------------------------------------------------
# Keras Classfier
# ------------------------------------------------------
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import cross_val_score
#from math import sqrt
#
#seed = 7
#np.random.seed(seed)
#
#def piledLayers():
#    return pileLayers(X.shape[1], optimizer, loss, layers)
#
#kfold = StratifiedKFold(n_splits=40, shuffle=True, random_state=seed)
#classifier = KerasClassifier(build_fn=piledLayers, batch_size=batchSize, nb_epoch=epochs)
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = kfold, n_jobs=-1, scoring=cross_val_scoring_type)
#accuracy = accuracies.mean()
#
#
#cross_val_scores_csv_path = write_path + 'past_jobs/'  + 'cross_val_scoress.csv'            
#if not os.path.exists(os.path.dirname(cross_val_scores_csv_path)):
#    try:
#        os.makedirs(os.path.dirname(saved_parameter_file))
#    except OSError as exc: # Guard against race condition
#        if exc.errno != errno.EEXIST:
#            raise
#    if not os.path.isfile(cross_val_scores_csv_path):
#        with open(cross_val_scores_csv_path, 'w') as f:
#            f.write('cross_val_score,jobId,date,\n')            
#
#with open(cross_val_scores_csv_path, 'a') as f:
#    f.write(str(accuracy) + ',' + id + ',' + str(datetime.datetime.now()) + '\n')
#
#if cross_val_only:
#    result = {'status': 200, 
#          'message': 'Cross validation complete.', 
#          'scores': str(accuracy) 
#          }
#    print(str(json.dumps(result)))
#    quit()

# ------------------------------------------------------
# NN Execute
# ------------------------------------------------------
history = model.fit(X_train, y_train, epochs = epochs, batch_size = batchSize, callbacks=callbacks_list)
exe_count + 1
## ------------------------------------------------------
# Create a dataframe from prediction and test
# ------------------------------------------------------
from math import sqrt
y_pred = model.predict(X_test)

y_pred = y_pred.astype('int64')
y_pred = pd.Series(y_pred.flatten().tolist())
y_test = pd.Series(y_test.tolist())

from sklearn.metrics import mean_squared_error
compare = pd.concat([y_pred, y_test, (abs(y_pred-y_test))], axis = 1)
compare.columns = ['y_pred', 'y_test', 'abs_dff']
compare.to_csv(projectPath + '/compare_' + id + '.csv')
rmse = sqrt(mean_squared_error(compare['y_pred'], compare['y_test']))

# ------------------------------------------------------
# Plot Training History
# ------------------------------------------------------
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot
pyplot.clf()
pyplot.plot(history.history['mean_squared_error'])
pyplot.suptitle(projectName + " " + id)
pyplot.title('RMSE: ' + str(rmse))
pyplot.savefig(projectPath + "/training" + "_run" + str(exe_count)  + ".pdf")
pyplot.clf()

# ------------------------------------------------------
# Plot Accuracy and Variance
# ------------------------------------------------------
pyplot.clf()
pyplot.figure(1)
pyplot.ylim(0, compare['y_test'].max())
pyplot.xlim(0, compare['y_test'].max())
pyplot.scatter(compare['y_test'], compare['y_pred'], c='red', alpha=0.05)
pyplot.xlabel('Actual', fontsize=18)
pyplot.ylabel('Predicted', fontsize=16)
pyplot.suptitle(projectName + " " + id)
pyplot.title('RMSE: ' + str(rmse))
pyplot.savefig(projectPath + "/accuracy_variance" + "_run" + str(exe_count)  + ".pdf")

# ------------------------------------------------------
# Save results
# ------------------------------------------------------
with open(trainingInfoFilePath,"a") as f:
    f.write("Mean error in days: " + str(compare['abs_dff'].mean()))
    f.close()

model.save(rootFolder + id + "_" +"weights.hdf5")

return_model = str(model.to_json())

scores = model.evaluate(X_test, y_test, batch_size=500)
result = {'status': 200, 
          'message': 'NN Training Complete', 
          'weightsPath': 'test',
          'scores': str(scores),
          'model': return_model }
print(str(json.dumps(result)))
