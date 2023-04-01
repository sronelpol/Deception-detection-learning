#!/usr/bin/env python
# coding: utf-8

# ## A Machine Learning Approach to combine all three modalities i.e. Audio, Micro-Expression and Gaze
# 
# Audio, gaze and Microexpression is combined using a Classifier and importance for each modality is plotted

# In[226]:


import sys

sys.path.append('/home/adrikamukherjee/venv/lib/python3.6/site-packages')

# In[1]:


# import necessary packages
import os, sys, glob, csv, numpy as np, pandas as pd, matplotlib.pyplot as plt, tensorflow as tf, keras
from sklearn import model_selection, preprocessing
from os import walk, path
from keras import models, layers, optimizers, preprocessing as KRSpreps, utils as KRSutils
import xgboost as xgb
from keras import utils as np_utils
import re
from sklearn import metrics
import matplotlib.pyplot as plt
import glob
import datetime
from sklearn.model_selection import train_test_split

# ### Create Dictionary and store paths for all different Modalities(Audio, Micro-expression and Gaze)

# In[228]:


data_path = {}
data_path['audiodata_path'] = "/home/adrikamukherjee/Audio_Features/csv_full_audio"
data_path['gazedata_path'] = "/home/adrikamukherjee/Gaze_Features"
data_path['mexpdata_path'] = "/home/adrikamukherjee/Mexp_Features"

# ### Checking No. of files in each of Audio, Micro-expression & Gaze Folders && Shape of the Dataframes

# In[229]:


c_gaze = 0
c_mexp = 0
c_audio = 0
for gazefilepath in glob.glob(os.path.join(data_path['audiodata_path'], '*.csv')):
    c_gaze += 1
for audiofilepath in glob.glob(os.path.join(data_path['gazedata_path'], '*.csv')):
    c_audio += 1
for mexpdatapath in glob.glob(os.path.join(data_path['mexpdata_path'], '*.csv')):
    c_mexp += 1

print("Number of Gaze Data : " + str(c_gaze))
print("Number of Audio Data : " + str(c_audio))
print("Number of Mexp Data : " + str(c_mexp))

# ## Creating Dictionaries of Audio, Micro-expression & Gaze
# Remove Initials and Make the Keys Same for the Same data

# In[230]:


from glob import glob

# In[231]:


data_shape_all = pd.DataFrame()
for key in data_path.keys():
    count = 0
    data_shape, file_names = [], []
    for filepath in glob(os.path.join(data_path[key], '*.csv')):
        file_shape = pd.read_csv(filepath).shape
        data_shape.append([file_shape[0], file_shape[1]])
        filename = path.basename(filepath)
        for reps in (("Gaze_", ""), ("Audio_", ""), ("Mexp_", "")):
            filename = filename.replace(*reps)
        file_names.append(filename)
        count += 1
    data_shape = pd.DataFrame(data_shape)
    data_shape.columns = [key + str(0), key + str(1)]
    data_shape.index = pd.Series(file_names)
    data_shape_all = pd.concat([data_shape_all, data_shape], axis=1)
    print(f"No. of file in {key}: ", count)
data_shape_all

# In[249]:


audio_dict, gaze_dict, mexp_dict = {}, {}, {}
listofdicts = [audio_dict, gaze_dict, mexp_dict]
for key, data_dict_indiv in zip(data_path.keys(), listofdicts):
    for filepath in glob(path.join(data_path[key], '*.csv')):
        data = pd.read_csv(filepath)
        filename = path.basename(filepath)
        for reps in (("Gaze_", ""), ("Audio_", ""), ("Mexp_", "")):
            filename = filename.replace(*reps)
        data_dict_indiv[filename] = data

# ### Checking If the Labels are Same for Same Keys in Each Dcitionaries & Separating Labels from Training Data

# In[250]:


filename_dictkeys = list(audio_dict)
label_dict = {}
for key in filename_dictkeys:
    audiodata, gazedata, mexpdata = audio_dict[key], gaze_dict[key], mexp_dict[key]
    label_audio = audiodata.loc[:, "label"].unique()[0]
    label_gaze = gazedata.loc[:, "label"].unique()[0]
    label_mexp = mexpdata.loc[:, "label"].unique()[0]
    label_set = set([label_audio, label_gaze, label_mexp])
    if len(label_set) > 1:
        print(key)
    else:
        label_dict[key] = list(label_set)[0]

# ### Encode labels and store it in a dictionary with the universal keys that holds all the data

# In[ ]:


label_to_num = lambda value: 0 if value == 'Truthful' else 1
label_dict_num = {key: label_to_num(value) for key, value in label_dict.items()}
label_dict_num

# ### Mean vector for all the modalities is calculated
# 
# frame wise data is replaced by single row which can be efficiently handled by Machine Learning Models

# In[252]:


df_list_mexp, df_list_audio, df_list_gaze, df_label = [], [], [], []
for key in filename_dictkeys:
    audio_dict[key].drop(["name", "frameTime", "Unnamed: 0", "label"], axis=1, inplace=True)
    audio_dict[key] = audio_dict[key].mean(axis=0)
    gaze_dict[key].drop(["frame", "Unnamed: 0", "label"], axis=1, inplace=True)
    gaze_dict[key] = gaze_dict[key].mean(axis=0)
    mexp_dict[key].drop(["frame", "Unnamed: 0", "label"], axis=1, inplace=True)
    mexp_dict[key] = mexp_dict[key].mean(axis=0)
    mexp_individual = mexp_dict[key].to_numpy()
    audio_individual = audio_dict[key].to_numpy()
    gaze_individual = gaze_dict[key].to_numpy()
    label_individual = label_dict_num[key]
    df_list_mexp.append(mexp_individual)
    df_list_audio.append(audio_individual)
    df_list_gaze.append(gaze_individual)
    df_label.append(label_individual)

# ### Convert Dataframe to numpy arrays

# In[253]:


df_list_mexp = np.asarray(df_list_mexp)
df_list_mexp.shape
df_list_audio = np.asarray(df_list_audio)
df_list_audio.shape
df_list_gaze = np.asarray(df_list_gaze)
df_list_gaze.shape

# ### Split train and test data && check if the labels are aligned

# In[260]:


split1 = train_test_split(df_list_mexp, df_label, test_size=0.1, random_state=42)
split2 = train_test_split(df_list_audio, df_label, test_size=0.1, random_state=42)
split3 = train_test_split(df_list_gaze, df_label, test_size=0.1, random_state=42)
(X_train_m, X_test_m, y_train_m, y_test_m) = split1
(X_train_a, X_test_a, y_train_a, y_test_a) = split2
(X_train_g, X_test_g, y_train_g, y_test_g) = split3

print(np.all(y_test_m == y_test_a))
print(np.all(y_test_a == y_test_g))

# ### XGboost is applied on the Audio data

# In[294]:


from xgboost.sklearn import XGBClassifier

model_a = XGBClassifier(silent=False,
                        scale_pos_weight=1,
                        learning_rate=0.01,
                        colsample_bytree=0.4,
                        subsample=0.8,
                        objective='binary:logistic',
                        n_estimators=1000,
                        reg_alpha=0.3,
                        max_depth=3,
                        gamma=10)
eval_set = [(X_train_a, y_train_a), (X_test_a, y_test_a)]
model_a.fit(X_train_a, y_train_a, early_stopping_rounds=10, eval_metric=["error", "logloss"], eval_set=eval_set,
            verbose=True)
# make predictions for test data
y_pred = model_a.predict(X_test_a)
predictions_audio_test = [round(value) for value in y_pred]
# evaluate predictions
accuracy = metrics.accuracy_score(y_test_a, predictions_audio_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# retrieve performance metrics
results = model_a.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()
# plot classification error
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.show()

# ### Display the Classification Report

# In[295]:


from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, \
    f1_score

print('Accuracy:', accuracy_score(y_test_a, predictions_audio_test))
print('F1 score:', f1_score(y_test_a, predictions_audio_test))
print('Recall:', recall_score(y_test_a, predictions_audio_test))
print('Precision:', precision_score(y_test_a, predictions_audio_test))
print('\n clasification report:\n', classification_report(y_test_a, predictions_audio_test))
print('\n confussion matrix:\n', confusion_matrix(y_test_a, predictions_audio_test))

# ###  XGboost is applied on gaze data

# In[296]:


from xgboost.sklearn import XGBClassifier

model_g = XGBClassifier(silent=False,
                        scale_pos_weight=1,
                        learning_rate=0.01,
                        colsample_bytree=0.4,
                        subsample=0.8,
                        objective='binary:logistic',
                        n_estimators=1000,
                        reg_alpha=0.3,
                        max_depth=3,
                        gamma=10)
eval_set = [(X_train_g, y_train_g), (X_test_g, y_test_g)]
model_g.fit(X_train_g, y_train_g, early_stopping_rounds=10, eval_metric=["error", "logloss"], eval_set=eval_set,
            verbose=True)
# make predictions for test data
y_pred = model_g.predict(X_test_g)
predictions_gaze_test = [round(value) for value in y_pred]
# evaluate predictions
accuracy = metrics.accuracy_score(y_test_g, predictions_gaze_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# retrieve performance metrics
results = model_g.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()
# plot classification error
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.show()

# ### Display the Metric Scores and Classification Report

# In[297]:


from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, \
    f1_score

print('Accuracy:', accuracy_score(y_test_g, predictions_gaze_test))
print('F1 score:', f1_score(y_test_g, predictions_gaze_test))
print('Recall:', recall_score(y_test_g, predictions_gaze_test))
print('Precision:', precision_score(y_test_g, predictions_gaze_test))
print('\n clasification report:\n', classification_report(y_test_g, predictions_gaze_test))
print('\n confussion matrix:\n', confusion_matrix(y_test_g, predictions_gaze_test))

# ### XGboost is applied on Micro_Expression data

# In[298]:


from xgboost.sklearn import XGBClassifier

model_m = XGBClassifier(silent=False,
                        scale_pos_weight=1,
                        learning_rate=0.01,
                        colsample_bytree=0.4,
                        subsample=0.8,
                        objective='binary:logistic',
                        n_estimators=1000,
                        reg_alpha=0.3,
                        max_depth=3,
                        gamma=10)
eval_set = [(X_train_m, y_train_m), (X_test_m, y_test_m)]
model_m.fit(X_train_m, y_train_m, early_stopping_rounds=10, eval_metric=["error", "logloss"], eval_set=eval_set,
            verbose=True)
# make predictions for test data
y_pred = model_m.predict(X_test_m)
predictions_mexp_test = [round(value) for value in y_pred]
# evaluate predictions
accuracy = metrics.accuracy_score(y_test_m, predictions_mexp_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# retrieve performance metrics
results = model_m.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()
# plot classification error
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.show()

# ### Display Classification Report of the above model

# In[299]:


from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, \
    f1_score

print('Accuracy:', accuracy_score(y_test_m, predictions_mexp_train))
print('F1 score:', f1_score(y_test_m, predictions_mexp_test))
print('Recall:', recall_score(y_test_m, predictions_mexp_test))
print('Precision:', precision_score(y_test_m, predictions_mexp_test))
print('\n clasification report:\n', classification_report(y_test_m, predictions_mexp_test))
print('\n confussion matrix:\n', confusion_matrix(y_test_m, predictions_mexp_train))

# ## Manually tuning hyper-parameters and assigning weights to different modalities during Late Fusion

# ### COMBINE MEXP+GAZE+AUDIO

# In[275]:


h_a = 0.5
h_m = 0.5
h_g = 0.3

y_pred_a = model_a.predict(X_test_a)
y_pred_m = model_m.predict(X_test_m)
y_pred_g = model_g.predict(X_test_g)

final_pred = h_a * y_pred_a + h_m * y_pred_m + h_g * y_pred_g
final_pred = [round(value) for value in final_pred]
if (y_test_m == y_test_a):
    if (y_test_a == y_test_g):
        y_test = y_test_a
print('Accuracy:', accuracy_score(y_test, final_pred))
print('F1 score:', f1_score(y_test, final_pred))
print('Recall:', recall_score(y_test, final_pred))
print('Precision:', precision_score(y_test, final_pred))
print('\n clasification report:\n', classification_report(y_test, final_pred))
print('\n confussion matrix:\n', confusion_matrix(y_test, final_pred))

# ### COMBINE MEXP+GAZE

# In[279]:


h_m = 0.7
h_g = 0.1

y_pred_m = model_m.predict(X_test_m)
y_pred_g = model_g.predict(X_test_g)

final_pred = h_m * y_pred_m + h_g * y_pred_g
final_pred = [round(value) for value in final_pred]
if (y_test_m == y_test_g):
    y_test = y_test_g
print('Accuracy:', accuracy_score(y_test, final_pred))
print('F1 score:', f1_score(y_test, final_pred))
print('Recall:', recall_score(y_test, final_pred))
print('Precision:', precision_score(y_test, final_pred))
print('\n clasification report:\n', classification_report(y_test, final_pred))
print('\n confussion matrix:\n', confusion_matrix(y_test, final_pred))

# ### COMBINE GAZE+AUDIO

# In[288]:


h_a = 0.8
h_g = 0.4

y_pred_a = model_a.predict(X_test_a)
y_pred_g = model_g.predict(X_test_g)

final_pred = h_a * y_pred_a + h_g * y_pred_g
final_pred = [round(value) for value in final_pred]

if (y_test_a == y_test_g):
    y_test = y_test_a
print('Accuracy:', accuracy_score(y_test, final_pred))
print('F1 score:', f1_score(y_test, final_pred))
print('Recall:', recall_score(y_test, final_pred))
print('Precision:', precision_score(y_test, final_pred))
print('\n clasification report:\n', classification_report(y_test, final_pred))
print('\n confussion matrix:\n', confusion_matrix(y_test, final_pred))

# ### COMBINE MEXP+AUDIO

# In[292]:


h_a = 0.5
h_m = 0.6

y_pred_a = model_a.predict(X_test_a)
y_pred_m = model_m.predict(X_test_m)

final_pred = h_a * y_pred_a + h_m * y_pred_m
final_pred = [round(value) for value in final_pred]
if (y_test_m == y_test_a):
    y_test = y_test_a
print('Accuracy:', accuracy_score(y_test, final_pred))
print('F1 score:', f1_score(y_test, final_pred))
print('Recall:', recall_score(y_test, final_pred))
print('Precision:', precision_score(y_test, final_pred))
print('\n clasification report:\n', classification_report(y_test, final_pred))
print('\n confussion matrix:\n', confusion_matrix(y_test, final_pred))

# ## Using Classifier model to vote for the best Modality
# 
# Predicted outputs from individual models are further processed by another XGBClassifier

# In[314]:


y_pred_a_test = model_a.predict(X_test_a)
y_pred_m_test = model_m.predict(X_test_m)
y_pred_g_test = model_g.predict(X_test_g)
y_pred_a_train = model_a.predict(X_train_a)
y_pred_m_train = model_m.predict(X_train_m)
y_pred_g_train = model_g.predict(X_train_g)
dfObj1 = pd.DataFrame(y_pred_a_train, columns=['Audio'])
dfObj2 = pd.DataFrame(y_pred_m_train, columns=['Mexp'])
dfObj3 = pd.DataFrame(y_pred_g_train, columns=['Gaze'])
dfObj4 = pd.DataFrame(y_pred_a_test, columns=['Audio'])
dfObj5 = pd.DataFrame(y_pred_m_test, columns=['Mexp'])
dfObj6 = pd.DataFrame(y_pred_g_test, columns=['Gaze'])
df_train_data = pd.concat([dfObj1, dfObj2, dfObj3], axis=1)
df_test_data = pd.concat([dfObj4, dfObj5, dfObj6], axis=1)
if (y_test_m == y_test_a):
    if (y_test_a == y_test_g):
        y_test = y_test_a
if (y_train_m == y_train_a):
    if (y_train_a == y_train_g):
        y_train = y_train_a
model = XGBClassifier(silent=False,
                      scale_pos_weight=1,
                      learning_rate=0.01,
                      colsample_bytree=0.4,
                      subsample=0.8,
                      objective='binary:logistic',
                      n_estimators=1000,
                      reg_alpha=0.3,
                      max_depth=3,
                      gamma=10)
eval_set = [(df_train_data, y_train), (df_test_data, y_test)]
model.fit(df_train_data, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, early_stopping_rounds=10,
          verbose=True)
y_pred = model.predict(df_test_data)
final_pred = [round(value) for value in y_pred]
print('Accuracy:', accuracy_score(y_test, final_pred))
print('F1 score:', f1_score(y_test, final_pred))
print('Recall:', recall_score(y_test, final_pred))
print('Precision:', precision_score(y_test, final_pred))
print('\n clasification report:\n', classification_report(y_test, final_pred))
print('\n confussion matrix:\n', confusion_matrix(y_test, final_pred))

# ### Display the Feature Importance for the modalities && Print weights given to each modality by the XGBClassifier

# In[319]:


from matplotlib import pyplot
from xgboost import plot_importance

# feature importance
print(model.feature_importances_)
plot_importance(model)
pyplot.show()

# ### RandomForestClassifier is used to vote for the feature Importance

# In[320]:


from sklearn.ensemble import RandomForestClassifier

y_pred_a_test = model_a.predict(X_test_a)
y_pred_m_test = model_m.predict(X_test_m)
y_pred_g_test = model_g.predict(X_test_g)
y_pred_a_train = model_a.predict(X_train_a)
y_pred_m_train = model_m.predict(X_train_m)
y_pred_g_train = model_g.predict(X_train_g)
dfObj1 = pd.DataFrame(y_pred_a_train, columns=['Audio'])
dfObj2 = pd.DataFrame(y_pred_m_train, columns=['Mexp'])
dfObj3 = pd.DataFrame(y_pred_g_train, columns=['Gaze'])
dfObj4 = pd.DataFrame(y_pred_a_test, columns=['Audio'])
dfObj5 = pd.DataFrame(y_pred_m_test, columns=['Mexp'])
dfObj6 = pd.DataFrame(y_pred_g_test, columns=['Gaze'])
df_train_data = pd.concat([dfObj1, dfObj2, dfObj3], axis=1)
df_test_data = pd.concat([dfObj4, dfObj5, dfObj6], axis=1)
if (y_test_m == y_test_a):
    if (y_test_a == y_test_g):
        y_test = y_test_a
if (y_train_m == y_train_a):
    if (y_train_a == y_train_g):
        y_train = y_train_a
model_randomforest = RandomForestClassifier(max_depth=2, random_state=0)
model_randomforest.fit(df_train_data, y_train)
y_pred = model_randomforest.predict(df_test_data)
final_pred = [round(value) for value in y_pred]
print('Accuracy:', accuracy_score(y_test, final_pred))
print('F1 score:', f1_score(y_test, final_pred))
print('Recall:', recall_score(y_test, final_pred))
print('Precision:', precision_score(y_test, final_pred))
print('\n clasification report:\n', classification_report(y_test, final_pred))
print('\n confussion matrix:\n', confusion_matrix(y_test, final_pred))

# ### Print weights given to each modality by the RandomForestClassifier

# In[321]:


print(model_randomforest.feature_importances_)

# ### Feature Importance obtained from RandomForestClassifier is plotted

# In[322]:


features = ['Audio', 'Mexp', 'Gaze']
importances = model_randomforest.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# In[ ]:
