import glob
import os

import pandas as pd

PRINT = True


def show_data(feature_name_to_path):
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


def count_number_of_files_per_model(feature_name_to_path=None):
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


def remove_feature_name_annotation(feature_name_to_path):
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
    filenames = list(audio_dict)
    filename_to_label = {}
    for filename in filenames:
        audio_data, gaze_data, micro_expressions_data = audio_dict[filename], gaze_dict[filename], mexp_dict[filename]
        label_audio = audio_data.loc[:, "label"].unique()[0]
        label_gaze = gaze_data.loc[:, "label"].unique()[0]
        label_mexp = micro_expressions_data.loc[:, "label"].unique()[0]
        labels = {label_audio, label_gaze, label_mexp}
        if len(labels) > 1:
            if PRINT:
                print(filename)
            raise ValueError
        else:
            filename_to_label[filename] = list(labels)[0]
    return filename_to_label, filenames


def encode_labels(filename_to_label):
    filename_to_encoded_label = {key: 0 if value == "Truthful" else 1 for key, value in filename_to_label.items()}
    if PRINT:
        print(filename_to_encoded_label)
    return filename_to_encoded_label
