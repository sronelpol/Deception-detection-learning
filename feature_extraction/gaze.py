import glob
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from feature_extraction.utils import run_open_face_in_docker

DIR = Path(__file__).parent.parent
DATASETS_DIR = DIR / "datasets"
RESOURCES_DIR = DIR / "resources"
GAZE_FEATURES_DIR = RESOURCES_DIR / "gaze_features"


def add_trial_testimonies_data():
    video_list = []
    dir_list = [
        f"{str(DATASETS_DIR)}/trial_testimonies/Deceptive",
        f"{str(DATASETS_DIR)}/trial_testimonies/Truthful"
    ]
    count_trial = 0
    for dir in dir_list:
        for filename in glob.glob(os.path.join(dir, '*.mp4')):
            video_list.append(filename)
            count_trial += 1
    print("number of videos from trial data")
    print(count_trial)
    return video_list


def extract_gaze_features(videolist):
    dict_input_output = {}
    output_filename_list = list()
    gaze_trial = 0
    for filename in videolist:
        new_filename = os.path.split(filename)[-1]
        output_filename = "gaze_real_life_deception_" + new_filename
        output_filename = output_filename.replace(".mp4", ".csv")
        output_file_path = f"{str(GAZE_FEATURES_DIR)}/{output_filename}"
        run_open_face_in_docker(filename, output_file_path)
        gaze_trial += 1
        dict_input_output[filename] = output_file_path
        output_filename_list.append(output_file_path)
    print("converted wav files counts are:")
    print("Trail = " + str(gaze_trial))
    return dict_input_output, output_filename_list


def annotate(dict_input_output):
    head = []
    head.append("Path_for_mp4_video")
    head.append("csv_file_name")
    head.append("csv_file_name_path_mexp_data")
    head.append("label")
    index = 0
    df_input_output = pd.DataFrame(columns=head)
    for key, value in dict_input_output.items():
        df_input_output = df_input_output.append(pd.Series(np.nan, index=head), ignore_index=True)
        df_input_output.iloc[index, head.index('Path_for_mp4_video')] = key
        df_input_output.iloc[index, head.index('csv_file_name_path_mexp_data')] = value
        csv_file_name = os.path.basename(value)
        df_input_output.iloc[index, head.index('csv_file_name')] = csv_file_name
        if "lie" in value:
            df_input_output.iloc[index, head.index('label')] = "Deceptive"
        if "truth" in value:
            df_input_output.iloc[index, head.index('label')] = "Truthful"
        index += 1

    print(len(df_input_output.index))
    df_input_output.to_csv(f"{str(GAZE_FEATURES_DIR)}/annotation_gaze_features.csv", index=False)


def re_annotate_something():
    df_readannotation = pd.read_csv(f"{str(GAZE_FEATURES_DIR)}/annotation_gaze_features.csv")
    dir = str(GAZE_FEATURES_DIR)
    value = None
    out_path = None
    for filename in glob.glob(os.path.join(dir, '*.csv')):
        file = os.path.basename(filename)
        df_readindividual = pd.read_csv(filename)
        for index, row in df_readannotation.iterrows():
            if (row["csv_file_name"] == file):
                value = row["label"]
                out_path = row["csv_file_name_path_gaze_data"]
        if value:
            df_readindividual["label"] = value
        if out_path:
            df_readindividual.to_csv(out_path)


if __name__ == '__main__':
    video_list = add_trial_testimonies_data()
    dict_input_output, output_filename_list = extract_gaze_features(video_list)
    annotate(dict_input_output)
