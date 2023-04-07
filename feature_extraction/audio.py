import glob
import os
from pathlib import Path

import pandas as pd

from feature_extraction.arff_to_csv import convert_arrf_file_to_csv
from feature_extraction.helpers import add_trial_testimonies_data
from feature_extraction.utils import count_number_of_wav_files, annotate_audio_files, extract_audio_features, \
    convert_to_mp4_to_wav_files

DIR = Path(__file__).parent.parent
DATASETS_DIR = DIR / "datasets"
RESOURCES_DIR = DIR / "resources"
WAV_DIR = RESOURCES_DIR / "wavfiles"
AUDIO_FEATURES_DIR = RESOURCES_DIR / "audio_features"


def re_annotate_and_combine_csvs():
    df_read_annotation = pd.read_csv("annotation_audio_features.csv")
    df_combined = pd.DataFrame()
    dir = f"{str(AUDIO_FEATURES_DIR)}/arff_files_full_video"
    for filename in glob.glob(os.path.join(dir, '*.csv')):
        file = os.path.basename(filename)
        df_readindividual = pd.read_csv(filename)
        value = False
        outputpath = False
        for index, row in df_read_annotation.iterrows():
            if row["csv_file_name"] == file:
                value = row["label"]
                outputpath = row["csv_file_name_path"]
        if not value:
            raise ValueError
        df_readindividual["label"] = value
        # del df_readindividual['emotion']
        if not outputpath:
            raise ValueError
        df_readindividual.to_csv(outputpath)
        df_combined = df_combined.append(df_readindividual, ignore_index=True)
    combined_csv_path = dir + "/Combined_csv_fullvideo.csv"
    df_combined.to_csv(combined_csv_path)


def copy_all_csvs_to_destination_folder():
    list_dir = [f"{str(AUDIO_FEATURES_DIR)}/arff_files_full_video"]
    out_dir = f"{str(AUDIO_FEATURES_DIR)}/csv_full_audio"
    for dir in list_dir:
        for filename in glob.glob(os.path.join(dir, '*.csv')):
            file = os.path.basename(filename)
            out_path = out_dir + "/" + file
            cmd = "cp " + filename + " " + out_path
            os.system(cmd)


if __name__ == '__main__':
    video_list = add_trial_testimonies_data()
    dict_input_output, output_filename_list = convert_to_mp4_to_wav_files(video_list,
                                                                          output_dir=f"{str(RESOURCES_DIR)}/wavfiles/",
                                                                          prefix_name="audio_real_life_deception_")
    count_number_of_wav_files(str(RESOURCES_DIR / "wavfiles"))
    annotate_audio_files(dict_input_output, dir_path=f"{str(RESOURCES_DIR)}/audio_features/arff_files_full_video/")
    extract_audio_features(input_dir=WAV_DIR, output_dir=f"{AUDIO_FEATURES_DIR}/arff_files_full_video/")
    convert_arrf_file_to_csv(f"{str(AUDIO_FEATURES_DIR)}/arff_files_full_video")
    re_annotate_and_combine_csvs()
    # copy_all_csvs_to_destination_folder()
    # use cp as  the copy function is not working on windows:  cp *.csv ..\csv_full_audio
