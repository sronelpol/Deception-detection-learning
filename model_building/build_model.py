import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DIR = Path(__file__).parent.parent
DATASETS_DIR = DIR / "datasets"
RESOURCES_DIR = DIR / "resources"
AUDIO_FEATURES_DIR = RESOURCES_DIR / "audio_features" / "csv_full_audio"
GAZE_FEATURES_DIR = RESOURCES_DIR / "gaze_features"
MICRO_EXPRESSION_FEATURES_DIR = RESOURCES_DIR / "micro_expression_features"

FEATURE_NAME_TP_PATH = {
    'audio_data': str(AUDIO_FEATURES_DIR),
    'gaze_data': str(GAZE_FEATURES_DIR),
    'micro_expression_data': str(MICRO_EXPRESSION_FEATURES_DIR)
}

c_gaze = 0
c_mexp = 0
c_audio = 0
for gazefilepath in glob.glob(os.path.join(FEATURE_NAME_TP_PATH['gaze_data'], '*.csv')):
    c_gaze += 1
for audiofilepath in glob.glob(os.path.join(FEATURE_NAME_TP_PATH['audio_data'], '*.csv')):
    c_audio += 1
for mexpdatapath in glob.glob(os.path.join(FEATURE_NAME_TP_PATH['micro_expression_data'], '*.csv')):
    c_mexp += 1

print("Number of Gaze Data : " + str(c_gaze))
print("Number of Audio Data : " + str(c_audio))
print("Number of Mexp Data : " + str(c_mexp))

data_shape_all = pd.DataFrame()
for key in FEATURE_NAME_TP_PATH.keys():
    count = 0
    data_shape, file_names = [], []
    for filepath in glob.glob(os.path.join(FEATURE_NAME_TP_PATH[key], '*.csv')):
        file_shape = pd.read_csv(filepath).shape
        data_shape.append([file_shape[0], file_shape[1]])
        filename = os.path.basename(filepath)
        for reps in (("gaze_", ""), ("audio_", ""), ("micro_expressions_", "")):
            filename = filename.replace(*reps)
        file_names.append(filename)
        count += 1
    data_shape = pd.DataFrame(data_shape)
    data_shape.columns = [key + str(0), key + str(1)]
    data_shape.index = pd.Series(file_names)
    data_shape_all = pd.concat([data_shape_all, data_shape], axis=1)
    print(f"No. of file in {key}: ", count)
print(data_shape_all)

audio_dict, gaze_dict, mexp_dict = {}, {}, {}
listofdicts = [audio_dict, gaze_dict, mexp_dict]
for key, data_dict_indiv in zip(FEATURE_NAME_TP_PATH.keys(), listofdicts):
    for filepath in glob.glob(os.path.join(FEATURE_NAME_TP_PATH[key], '*.csv')):
        data = pd.read_csv(filepath)
        filename = os.path.basename(filepath)
        for reps in (("gaze_", ""), ("audio_", ""), ("micro_expressions_", "")):
            filename = filename.replace(*reps)
        data_dict_indiv[filename] = data

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

label_to_num = lambda value: 0 if value == 'Truthful' else 1
label_dict_num = {key: label_to_num(value) for key, value in label_dict.items()}
print(label_dict_num)

df_list_mexp, df_list_audio, df_list_gaze, df_label = [], [], [], []
for key in filename_dictkeys:
    audio_dict[key].drop(["name", "Unnamed: 0", "label"], axis=1, inplace=True)
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

df_list_mexp = np.asarray(df_list_mexp)
df_list_mexp.shape
df_list_audio = np.asarray(df_list_audio)
df_list_audio.shape
df_list_gaze = np.asarray(df_list_gaze)
df_list_gaze.shape

split1 = train_test_split(df_list_mexp, df_label, test_size=0.1, random_state=42)
split2 = train_test_split(df_list_audio, df_label, test_size=0.1, random_state=42)
split3 = train_test_split(df_list_gaze, df_label, test_size=0.1, random_state=42)
(X_train_m, X_test_m, y_train_m, y_test_m) = split1
(X_train_a, X_test_a, y_train_a, y_test_a) = split2
(X_train_g, X_test_g, y_train_g, y_test_g) = split3

print(np.all(y_test_m == y_test_a))
print(np.all(y_test_a == y_test_g))

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
                        gamma=10,
                        early_stopping_rounds=10,
                        eval_metric=["error", "logloss"],
                        )
eval_set = [(X_train_a, y_train_a), (X_test_a, y_test_a)]

model_a.fit(X_train_a, y_train_a, eval_set=eval_set, verbose=True)
# make predictions for test data
y_pred = model_a.predict(X_test_a)
predictions_audio_test = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test_a, predictions_audio_test)
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
