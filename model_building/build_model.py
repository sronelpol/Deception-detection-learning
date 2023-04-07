import glob
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import mode
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

FEATURE_NAME_TO_PATH = {
    "audio_data": str(AUDIO_FEATURES_DIR),
    "gaze_data": str(GAZE_FEATURES_DIR),
    "micro_expression_data": str(MICRO_EXPRESSION_FEATURES_DIR),
}
PRINT = True


def count_number_of_files_per_model(feature_name_to_path=None):
    if not feature_name_to_path:
        feature_name_to_path = FEATURE_NAME_TO_PATH
    count_gaze = 0
    count_micro_expressions = 0
    count_audio = 0
    for _ in glob.glob(os.path.join(feature_name_to_path["gaze_data"], "*.csv")):
        count_gaze += 1
    for _ in glob.glob(os.path.join(feature_name_to_path["audio_data"], "*.csv")):
        count_audio += 1
    for _ in glob.glob(os.path.join(feature_name_to_path["micro_expression_data"], "*.csv")):
        count_micro_expressions += 1

    print("Number of Gaze Data : " + str(count_gaze))
    print("Number of Audio Data : " + str(count_audio))
    print("Number of Micro Expressions Data : " + str(count_micro_expressions))


def show_data(feature_name_to_path=None):
    if not feature_name_to_path:
        feature_name_to_path = FEATURE_NAME_TO_PATH
    data_shape_all = pd.DataFrame()
    for key in feature_name_to_path.keys():
        count = 0
        data_shape, file_names = [], []
        for filepath in glob.glob(os.path.join(feature_name_to_path[key], "*.csv")):
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
        if PRINT:
            print(f"No. of file in {key}: ", count)
    if PRINT:
        print(data_shape_all)


def remove_feature_name_annotation(feature_name_to_path=None):
    if not feature_name_to_path:
        feature_name_to_path = FEATURE_NAME_TO_PATH
    audio, gaze, mexp = {}, {}, {}
    all_data = [audio, gaze, mexp]
    for key, data_individual in zip(feature_name_to_path.keys(), all_data):
        for filepath in glob.glob(os.path.join(feature_name_to_path[key], "*.csv")):
            data = pd.read_csv(filepath)
            filename = os.path.basename(filepath)
            for reps in (("gaze_", ""), ("audio_", ""), ("micro_expressions_", "")):
                filename = filename.replace(*reps)
            data_individual[filename] = data
    return audio, gaze, mexp


def check_and_prepare_labels(audio_dict, gaze_dict, mexp_dict):
    # Checking If the Labels are Same for Same Keys in Each dictionary & Separating Labels from Training Data
    _filenames = list(audio_dict)
    _filename_to_label = {}
    for filename in _filenames:
        audiodata, gazedata, mexpdata = audio_dict[filename], gaze_dict[filename], mexp_dict[filename]
        label_audio = audiodata.loc[:, "label"].unique()[0]
        label_gaze = gazedata.loc[:, "label"].unique()[0]
        label_mexp = mexpdata.loc[:, "label"].unique()[0]
        labels = {label_audio, label_gaze, label_mexp}
        if len(labels) > 1:
            if PRINT:
                print(filename)
            raise ValueError
        else:
            _filename_to_label[filename] = list(labels)[0]
    return _filename_to_label, _filenames


def encode_labels(_filename_to_label):
    _filename_to_encoded_label = {key: 0 if value == "Truthful" else 1 for key, value in _filename_to_label.items()}
    if PRINT:
        print(_filename_to_encoded_label)
    return _filename_to_encoded_label


def clean_and_prepare_data(audio, gaze, micro_expressions, label_to_number):
    # clean the data:
    df_micro_expressions, df_audio, df_gaze, df_label = [], [], [], []
    for key in filenames:
        audio[key].drop(["name", "Unnamed: 0", "label"], axis=1, inplace=True)
        audio[key] = audio[key].mean(axis=0, numeric_only=True)
        gaze[key].drop(["frame", "Unnamed: 0", "label"], axis=1, inplace=True)
        gaze[key] = gaze[key].mean(axis=0, numeric_only=True)
        micro_expressions[key].drop(["frame", "Unnamed: 0", "label"], axis=1, inplace=True)
        micro_expressions[key] = micro_expressions[key].mean(axis=0, numeric_only=True)
        micro_expressions_individual = micro_expressions[key].to_numpy()
        audio_individual = audio[key].to_numpy()
        gaze_individual = gaze[key].to_numpy()
        label_individual = label_to_number[key]
        df_micro_expressions.append(micro_expressions_individual)
        df_audio.append(audio_individual)
        df_gaze.append(gaze_individual)
        df_label.append(label_individual)
    df_micro_expressions = np.asarray(df_micro_expressions)
    if PRINT:
        print(df_micro_expressions.shape)
    df_audio = np.asarray(df_audio)
    if PRINT:
        print(df_audio.shape)
    df_gaze = np.asarray(df_gaze)
    if PRINT:
        print(df_gaze.shape)
    return df_micro_expressions, df_audio, df_gaze, df_label


def run_training(X_train, y_train, X_test, y_test, title="", should_print=PRINT):
    model = MLPClassifier(random_state=42, n_iter_no_change=20, verbose=True)
    history = model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions_audio_test = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions_audio_test)
    if should_print:
        print(f"{title} Accuracy: %.2f%%" % (accuracy * 100.0))
    # Calculate the log loss
    y_prob = model.predict_proba(X_test)
    logloss = log_loss(y_test, y_prob)
    if should_print:
        print("Log Loss: %.2f" % logloss)

    # Plot the loss and accuracy over epochs
    plt.plot(history.loss_curve_)  # noqa
    plt.title(f"Loss Curve {title}")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.show()

    joblib.dump(model, f"{str(MODELS_DIR)}/{title}_model.pkl")
    return model


def train_the_models_and_combine(should_print=PRINT):
    model_a = run_training(X_train_a, y_train_a, X_test_a, y_test_a, title="Audio")
    model_m = run_training(X_train_m, y_train_m, X_test_m, y_test_m, title="Micro Expressions")
    model_g = run_training(X_train_g, y_train_g, X_test_g, y_test_g, title="Gaze")

    # Make predictions using each model
    y_pred_a = model_a.predict(X_test_a)
    y_pred_m = model_m.predict(X_test_m)
    y_pred_g = model_g.predict(X_test_g)

    # Combine the predictions using majority voting
    y_pred = mode([y_pred_a, y_pred_m, y_pred_g], axis=0, keepdims=True)[0][0]

    # Calculate the accuracy of the combined predictions
    accuracy = accuracy_score(y_test_a, y_pred)
    if should_print:
        print("combined")
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
    joblib.dump(y_pred, f"{str(MODELS_DIR)}/combined_model.pkl")


if __name__ == "__main__":
    count_number_of_files_per_model()
    show_data()
    audio_data, gaze_data, micro_expressions_data = remove_feature_name_annotation()
    filename_to_label, filenames = check_and_prepare_labels(audio_data, gaze_data, micro_expressions_data)
    filename_to_encoded_label = encode_labels(filename_to_label)
    df_micro_expressions, df_audio, df_gaze, df_label = clean_and_prepare_data(
        audio_data, gaze_data, micro_expressions_data, filename_to_encoded_label
    )
    split1 = train_test_split(df_micro_expressions, df_label, test_size=0.1, random_state=42)
    split2 = train_test_split(df_audio, df_label, test_size=0.1, random_state=42)
    split3 = train_test_split(df_gaze, df_label, test_size=0.1, random_state=42)
    (X_train_m, X_test_m, y_train_m, y_test_m) = split1
    (X_train_a, X_test_a, y_train_a, y_test_a) = split2
    (X_train_g, X_test_g, y_train_g, y_test_g) = split3
    print(np.all(y_test_m == y_test_a))
    print(np.all(y_test_a == y_test_g))
    train_the_models_and_combine()
