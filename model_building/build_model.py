import glob
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


DIR = Path(__file__).parent.parent
DATASETS_DIR = DIR / "datasets"
RESOURCES_DIR = DIR / "resources"
MODELS_DIR = DIR / "models"

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

# Checking If the Labels are Same for Same Keys in Each dictionary & Separating Labels from Training Data
filename_dictkeys = list(audio_dict)
label_dict = {}
for key in filename_dictkeys:
    audiodata, gazedata, mexpdata = audio_dict[key], gaze_dict[key], mexp_dict[key]
    label_audio = audiodata.loc[:, "label"].unique()[0]
    label_gaze = gazedata.loc[:, "label"].unique()[0]
    label_mexp = mexpdata.loc[:, "label"].unique()[0]
    labels = {label_audio, label_gaze, label_mexp}
    if len(labels) > 1:
        print(key)
    else:
        label_dict[key] = list(labels)[0]

label_to_num = lambda value: 0 if value == 'Truthful' else 1
label_dict_num = {key: label_to_num(value) for key, value in label_dict.items()}
print(label_dict_num)

df_list_mexp, df_list_audio, df_list_gaze, df_label = [], [], [], []
for key in filename_dictkeys:
    audio_dict[key].drop(["name", "Unnamed: 0", "label"], axis=1, inplace=True)
    audio_dict[key] = audio_dict[key].mean(axis=0, numeric_only=True)
    gaze_dict[key].drop(["frame", "Unnamed: 0", "label"], axis=1, inplace=True)
    gaze_dict[key] = gaze_dict[key].mean(axis=0, numeric_only=True)
    mexp_dict[key].drop(["frame", "Unnamed: 0", "label"], axis=1, inplace=True)
    mexp_dict[key] = mexp_dict[key].mean(axis=0, numeric_only=True)
    mexp_individual = mexp_dict[key].to_numpy()
    audio_individual = audio_dict[key].to_numpy()
    gaze_individual = gaze_dict[key].to_numpy()
    label_individual = label_dict_num[key]
    df_list_mexp.append(mexp_individual)
    df_list_audio.append(audio_individual)
    df_list_gaze.append(gaze_individual)
    df_label.append(label_individual)

df_list_mexp = np.asarray(df_list_mexp)
print(df_list_mexp.shape)
df_list_audio = np.asarray(df_list_audio)
print(df_list_audio.shape)
df_list_gaze = np.asarray(df_list_gaze)
print(df_list_gaze.shape)
split1 = train_test_split(df_list_mexp, df_label, test_size=0.1, random_state=42)
split2 = train_test_split(df_list_audio, df_label, test_size=0.1, random_state=42)
split3 = train_test_split(df_list_gaze, df_label, test_size=0.1, random_state=42)
(X_train_m, X_test_m, y_train_m, y_test_m) = split1
(X_train_a, X_test_a, y_train_a, y_test_a) = split2
(X_train_g, X_test_g, y_train_g, y_test_g) = split3

print(np.all(y_test_m == y_test_a))
print(np.all(y_test_a == y_test_g))


def run_training(X_train, y_train, X_test, y_test, title=""):
    model = MLPClassifier(random_state=42, n_iter_no_change=20, verbose=True)
    history = model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions_audio_test = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions_audio_test)
    print(f"{title} Accuracy: %.2f%%" % (accuracy * 100.0))
    # Calculate the log loss
    y_prob = model.predict_proba(X_test)
    logloss = log_loss(y_test, y_prob)
    print("Log Loss: %.2f" % logloss)

    # Plot the loss and accuracy over epochs
    plt.plot(history.loss_curve_)
    plt.title(f'Loss Curve {title}')
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.show()


    joblib.dump(model, f"{str(MODELS_DIR)}/{title}_model.pkl")
    return model


model_a = run_training(X_train_a, y_train_a, X_test_a, y_test_a, title="Audio")
model_m = run_training(X_train_m, y_train_m, X_test_m, y_test_m, title="Micro Expressions")
model_g = run_training(X_train_g, y_train_g, X_test_g, y_test_g, title="Gaze")

from scipy.stats import mode

# Make predictions using each model
y_pred_a = model_a.predict(X_test_a)
y_pred_m = model_m.predict(X_test_m)
y_pred_g = model_g.predict(X_test_g)

# Combine the predictions using majority voting
y_pred = mode([y_pred_a, y_pred_m, y_pred_g], axis=0)[0][0]

# Calculate the accuracy of the combined predictions
accuracy = accuracy_score(y_test_a, y_pred)
print("combined")
print("Accuracy: %.2f%%" % (accuracy * 100.0))
joblib.dump(y_pred, f"{str(MODELS_DIR)}/combined_model.pkl")

