import os
import sys
import glob
import csv
from pathlib import Path
from pydub import AudioSegment

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras import models
from keras import layers
from os import walk
from os.path import splitext
from os.path import join

from resources.audio_features.arff_files_full_video.arffToCsv import main

DIR = Path(__file__).parent.parent
DATASETS_DIR = DIR / "datasets"
RESOURCES_DIR = DIR / "resources"
WAV_DIR = RESOURCES_DIR / "wavfiles"
AUDIO_FEATURES_DIR = RESOURCES_DIR / "audio_features"


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


def convert_to_wav_files(video_list, convert=False):
    exit_status = None
    # dictionary of input and output file path
    dict_input_output = {}
    output_filename_list = list()
    wav_trial = 0
    for filename in video_list:
        cmd = "ffmpeg -i " + "\"" + filename + "\"" + " "
        filename_with_file_type = os.path.split(filename)[-1]
        output_filename = "audio_real_life_deception_" + filename_with_file_type
        wav_trial += 1
        output_filename = output_filename.replace(".mp4", ".wav")
        outpath = f"{str(RESOURCES_DIR)}/wavfiles/" + output_filename
        cmd = cmd + outpath
        if convert:
            exit_status = os.system(cmd)
        if exit_status == 0 or not convert:
            dict_input_output[filename] = outpath
            output_filename_list.append(outpath)
    print("converted wav files counts are:")
    print("Trail = " + str(wav_trial))
    return dict_input_output, output_filename_list


def count_number_of_wav_files():
    count = 0
    wav_list = []
    for files in glob.glob(os.path.join(str(RESOURCES_DIR / "wavfiles"), '*.wav')):
        wav_list.append(files)
        count += 1
    print(count)


def annotate(dict_input_output, ):
    head = []
    head.append("Path_for_mp4_video")
    head.append("Path_for_wav_file")
    head.append("csv_file_name")
    head.append("csv_file_name_path_fullvideo")
    head.append("csv_file_name_path_perframe")
    head.append("label")
    dir1 = f"{str(RESOURCES_DIR)}/audio_features/arff_files_frame_wise/"
    dir2 = f"{str(RESOURCES_DIR)}/audio_features/arff_files_full_video/"

    indexi = 0
    df_input_output = pd.DataFrame(columns=head)
    for key, value in dict_input_output.items():
        df_input_output = df_input_output.append(pd.Series(np.nan, index=head), ignore_index=True)
        df_input_output.iloc[indexi, head.index('Path_for_mp4_video')] = key
        df_input_output.iloc[indexi, head.index('Path_for_wav_file')] = value
        csv_file_name = os.path.basename(value)
        csv_file_name = csv_file_name.replace(".wav", ".csv")
        df_input_output.iloc[indexi, head.index('csv_file_name')] = csv_file_name
        df_input_output.iloc[indexi, head.index('csv_file_name_path_fullvideo')] = dir2 + csv_file_name
        df_input_output.iloc[indexi, head.index('csv_file_name_path_perframe')] = dir1 + csv_file_name
        if "lie" in value:
            df_input_output.iloc[indexi, head.index('label')] = "Deceptive"
        if "truth" in value:
            df_input_output.iloc[indexi, head.index('label')] = "Truthful"

        indexi += 1
    print(len(df_input_output.index))
    df_input_output.to_csv("Annotation_audio_features.csv", index=False)


def extract_features_full_video():
    out_dir = f"{AUDIO_FEATURES_DIR}/arff_files_full_video/"
    # out_dir = f"{AUDIO_FEATURES_DIR}/arff_files_frame_wise/"
    wav_file_count = 0
    for filename in glob.glob(os.path.join(WAV_DIR, '*.wav')):
        input_filename = os.path.basename(filename)
        output_filename = input_filename.replace(".wav", ".arff")
        cmd = "/Users/ronelpoliak/Afeka/Final_Project/opensmile/build/progsrc/smilextract/SMILExtract -C /Users/ronelpoliak/Afeka/Final_Project/opensmile/config/emobase/emobase.conf -I " + f"{str(WAV_DIR)}/{input_filename} -O {out_dir}{output_filename}"
        x = os.system(cmd)
        if (x == 0):
            wav_file_count += 1
    print(wav_file_count)


def convert_arrf_file_to_csv(filename="arff_files_full_video"):
    main(f"{str(AUDIO_FEATURES_DIR)}/{filename}")


def re_annotate_and_combine_csvs():
    df_readannotation = pd.read_csv("Annotation_audio_features.csv")
    df_combined = pd.DataFrame()
    dir = f"{str(AUDIO_FEATURES_DIR)}/arff_files_full_video"
    for filename in glob.glob(os.path.join(dir, '*.csv')):
        file = os.path.basename(filename)
        df_readindividual = pd.read_csv(filename)
        for index, row in df_readannotation.iterrows():
            if (row["csv_file_name"] == file):
                val = row["label"]
                outputpath = row["csv_file_name_path_fullvideo"]
        df_readindividual["label"] = val
        # del df_readindividual['emotion']
        df_readindividual.to_csv(outputpath)
        df_combined = df_combined.append(df_readindividual, ignore_index=True)
    combined_csv_path = dir + "/Combined_csv_fullvideo.csv"
    df_combined.to_csv(combined_csv_path)


if __name__ == '__main__':
    # video_list = add_trial_testimonies_data()
    # dict_input_output, output_filename_list = convert_to_wav_files(video_list)
    # count_number_of_wav_files()
    # annotate(dict_input_output)
    # extract_features_full_video()
    # convert_arrf_file_to_csv()
    re_annotate_and_combine_csvs()
