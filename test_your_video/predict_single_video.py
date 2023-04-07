from pathlib import Path

import joblib
from scipy.stats import mode

from feature_extraction.arff_to_csv import convert_arrf_file_to_csv
from feature_extraction.audio import re_annotate_audio_data_and_combine_csvs
from feature_extraction.utils import (
    extract_open_face_features,
    annotate_openface_output,
    re_annotate_openface_output,
    convert_to_mp4_to_wav_files,
    count_number_of_wav_files,
    annotate_audio_files,
    extract_audio_features,
)
from model_building.build_model import clean_and_prepare_data
from model_building.utils import (
    count_number_of_files_per_model,
    show_data,
    remove_feature_name_annotation,
    check_and_prepare_labels,
    encode_labels,
    delete_unnecessary_files,
)

DIR = Path(__file__).parent.parent
RESOURCES_DIR = DIR / "resources"
MODELS_DIR = DIR / "models"
AUDIO_FEATURES_DIR = RESOURCES_DIR / "test_your_video" / "audio_features"
GAZE_FEATURES_DIR = RESOURCES_DIR / "test_your_video" / "gaze_features"
MICRO_EXPRESSION_FEATURES_DIR = RESOURCES_DIR / "test_your_video" / "micro_expression_features"

# TODO: put here the full path for your mp4 video
VIDEO_FULL_PATH = r"C:\Users\ronel\afeka\Deception-detection-learning\datasets\
youtube\can_you_tell_when_someone_is_lying_to_you_1_truth.mp4"
FEATURE_NAME_TO_PATH = {
    "audio_data": str(AUDIO_FEATURES_DIR),
    "gaze_data": str(GAZE_FEATURES_DIR),
    "micro_expression_data": str(MICRO_EXPRESSION_FEATURES_DIR),
}


def extract_data_features():
    # micro expressions features:
    dict_input_output, output_filename_list = extract_open_face_features(
        [VIDEO_FULL_PATH],
        path_dir=str(MICRO_EXPRESSION_FEATURES_DIR),
        prefix_name="micro_expressions_",
        mode=["pose", "aus"],
    )
    annotate_openface_output(dict_input_output, str(MICRO_EXPRESSION_FEATURES_DIR))
    re_annotate_openface_output(str(MICRO_EXPRESSION_FEATURES_DIR))
    # gaze features :
    dict_input_output, output_filename_list = extract_open_face_features(
        [VIDEO_FULL_PATH],
        path_dir=str(GAZE_FEATURES_DIR),
        prefix_name="gaze_",
        mode="gaze",
    )
    annotate_openface_output(dict_input_output, str(GAZE_FEATURES_DIR))
    re_annotate_openface_output(str(GAZE_FEATURES_DIR))
    # audio_features
    dict_input_output, output_filename_list = convert_to_mp4_to_wav_files(
        [VIDEO_FULL_PATH], output_dir=str(AUDIO_FEATURES_DIR), prefix_name="audio_"
    )
    count_number_of_wav_files(str(AUDIO_FEATURES_DIR))
    annotate_audio_files(dict_input_output, dir_path=str(AUDIO_FEATURES_DIR))
    extract_audio_features(input_dir=str(AUDIO_FEATURES_DIR), output_dir=str(AUDIO_FEATURES_DIR))
    convert_arrf_file_to_csv(str(AUDIO_FEATURES_DIR))
    re_annotate_audio_data_and_combine_csvs(str(AUDIO_FEATURES_DIR))


def prepare_data_for_assessment():
    delete_unnecessary_files(FEATURE_NAME_TO_PATH.values())
    count_number_of_files_per_model(FEATURE_NAME_TO_PATH)
    show_data(FEATURE_NAME_TO_PATH)
    audio_data, gaze_data, micro_expressions_data = remove_feature_name_annotation(FEATURE_NAME_TO_PATH)
    filename_to_label, filenames = check_and_prepare_labels(audio_data, gaze_data, micro_expressions_data)
    filename_to_encoded_label = encode_labels(filename_to_label)
    df_micro_expressions, df_audio, df_gaze, df_label = clean_and_prepare_data(
        audio_data, gaze_data, micro_expressions_data, filename_to_encoded_label, filenames
    )
    return df_micro_expressions, df_audio, df_gaze, df_label


def assess_model_with_youtube_data(df_audio, df_gaze):
    # Load the saved model
    model_a = joblib.load(f"{MODELS_DIR}/audio_model.pkl")
    joblib.load(f"{MODELS_DIR}/micro_expressions_model.pkl")
    model_g = joblib.load(f"{MODELS_DIR}/gaze_model.pkl")

    y_pred_a = model_a.predict(df_audio)
    # y_pred_m = model_m.predict(df_micro_expressions)
    y_pred_g = model_g.predict(df_gaze)

    # Combine the predictions using majority voting
    y_pred = mode([y_pred_a, y_pred_g], axis=0, keepdims=True)[0][0]
    # y_pred = mode([y_pred_a, y_pred_m, y_pred_g], axis=0, keepdims=True)[0][0]

    if y_pred:
        print("deception")
    else:
        print("truth")


if __name__ == "__main__":
    extract_data_features()
    _, df_audio, df_gaze, _ = prepare_data_for_assessment()
    assess_model_with_youtube_data(df_audio, df_gaze)
