import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).parent.parent  # main directory of the project (Deception-detection-learning)
# check that openface is located outside the project folder (but in the same "father" folder)
OPENFACE = DIR.parent / "OpenFace_2.2.0_win_x64" / "FeatureExtraction.exe"
# check that opensmile is located outside the project folder (but in the same "father" folder)
SMILE_EXTRACT = DIR.parent / "opensmile-3.0.1-win-x64" / "bin" / "SMILExtract.exe"
SMILE_CONFIG = DIR.parent / "opensmile-3.0.1-win-x64" / "config" / "emobase" / "emobase.conf"


def listify(something):
    if isinstance(something, list):
        return something
    return [something]


def extract_open_face_features(video_list, path_dir, mode, prefix_name):
    dict_input_output = {}
    output_filename_list = list()
    count = 0
    for filename in video_list:
        new_filename = os.path.split(filename)[-1]
        output_filename = prefix_name + new_filename
        output_filename = output_filename.replace(".mp4", ".csv")
        output_file_path = f"{path_dir}/{output_filename}"
        modes = " ".join(["-" + item for item in listify(mode)])
        cmd = f"{str(OPENFACE)} -f {filename} -of {output_file_path} {modes}"
        exit_status = os.system(cmd)
        if exit_status:
            raise Exception
        count += 1
        dict_input_output[filename] = output_file_path
        output_filename_list.append(output_file_path)
    print("converted wav files counts are:")
    print(count)
    return dict_input_output, output_filename_list


def annotate_openface_output(dict_input_output, path_to_store):
    head = ["Path_for_mp4_video", "csv_file_name", "csv_file_name_path", "label"]
    index = 0
    df_input_output = pd.DataFrame(columns=head)
    for key, value in dict_input_output.items():
        df_input_output = df_input_output.append(pd.Series(np.nan, index=head), ignore_index=True)
        df_input_output.iloc[index, head.index('Path_for_mp4_video')] = key
        df_input_output.iloc[index, head.index('csv_file_name_path_gaze_data')] = value
        df_input_output.iloc[index, head.index('csv_file_name')] = os.path.basename(value)
        if "lie" in value:
            df_input_output.iloc[index, head.index('label')] = "Deceptive"
        if "truth" in value:
            df_input_output.iloc[index, head.index('label')] = "Truthful"
        index += 1

    print(len(df_input_output.index))
    df_input_output.to_csv(f"{path_to_store}/annotation.csv", index=False)


def re_annotate_openface_output(dir_path):
    df_readannotation = pd.read_csv(f"{dir_path}/annotation.csv")
    value = None
    out_path = None
    for filename in glob.glob(os.path.join(dir_path, '*.csv')):
        file = os.path.basename(filename)
        df_readindividual = pd.read_csv(filename)
        for index, row in df_readannotation.iterrows():
            if (row["csv_file_name"] == file):
                value = row["label"]
                out_path = row["csv_file_name_path"]
        if value:
            df_readindividual["label"] = value
        if out_path:
            df_readindividual.to_csv(out_path)


def convert_to_mp4_to_wav_files(video_list, output_dir, prefix_name="audio_", convert=False):
    exit_status = None
    # dictionary of input and output file path
    dict_input_output = {}
    output_filename_list = list()
    wav_count = 0
    for filename in video_list:
        cmd = "ffmpeg -i " + "\"" + filename + "\"" + " "
        filename_with_file_type = os.path.split(filename)[-1]
        output_filename = prefix_name + filename_with_file_type
        wav_count += 1
        output_filename = output_filename.replace(".mp4", ".wav")
        outpath = output_dir + output_filename
        cmd = cmd + outpath
        if convert:
            exit_status = os.system(cmd)
        if exit_status == 0 or not convert:
            dict_input_output[filename] = outpath
            output_filename_list.append(outpath)
    print("converted wav files counts are:")
    print(wav_count)
    return dict_input_output, output_filename_list


def count_number_of_wav_files(dir_path):
    count = 0
    wav_list = []
    for files in glob.glob(os.path.join(dir_path, '*.wav')):
        wav_list.append(files)
        count += 1
    print(count)


def annotate_audio_files(input_output, dir_path):
    head = ["Path_for_mp4_video", "Path_for_wav_file", "csv_file_name", "csv_file_name_path", "label"]

    index = 0
    df_input_output = pd.DataFrame(columns=head)
    for key, value in input_output.items():
        df_input_output = df_input_output.append(pd.Series(np.nan, index=head), ignore_index=True)
        df_input_output.iloc[index, head.index('Path_for_mp4_video')] = key
        df_input_output.iloc[index, head.index('Path_for_wav_file')] = value
        csv_file_name = os.path.basename(value)
        csv_file_name = csv_file_name.replace(".wav", ".csv")
        df_input_output.iloc[index, head.index('csv_file_name')] = csv_file_name
        df_input_output.iloc[index, head.index('csv_file_name_path')] = dir_path + csv_file_name
        if "lie" in value:
            df_input_output.iloc[index, head.index('label')] = "Deceptive"
        if "truth" in value:
            df_input_output.iloc[index, head.index('label')] = "Truthful"

        index += 1
    print(len(df_input_output.index))
    df_input_output.to_csv("annotation_audio_features.csv", index=False)


def extract_audio_features(input_dir, output_dir):
    wav_file_count = 0
    for filename in glob.glob(os.path.join(input_dir, '*.wav')):
        input_filename = os.path.basename(filename)
        output_filename = input_filename.replace(".wav", ".arff")
        open_smile = f"{str(SMILE_EXTRACT)} -C {str(SMILE_CONFIG)}"
        cmd = f"{open_smile} -I {str(input_dir)}/{input_filename} -O {output_dir}{output_filename}"
        x = os.system(cmd)
        if x == 0:
            wav_file_count += 1
    print(wav_file_count)
