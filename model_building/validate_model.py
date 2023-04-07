from pathlib import Path

from sklearn.metrics import accuracy_score
from scipy.stats import mode

from feature_extraction.arff_to_csv import convert_arrf_file_to_csv
from feature_extraction.audio import re_annotate_audio_data_and_combine_csvs
from feature_extraction.helpers import add_youtube_data
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
)
import joblib

DIR = Path(__file__).parent.parent
RESOURCES_DIR = DIR / "resources"
MODELS_DIR = DIR / "models"
AUDIO_FEATURES_DIR = RESOURCES_DIR / "youtube" / "audio_features"
GAZE_FEATURES_DIR = RESOURCES_DIR / "youtube" / "gaze_features"
MICRO_EXPRESSION_FEATURES_DIR = RESOURCES_DIR / "youtube" / "micro_expression_features"

FILENAME_TO_LABEL = {
    "can_you_tell_when_someone_is_lying_to_you_1_truth.mp4": "Truthful",
    "can_you_tell_when_someone_is_lying_to_you_2_truth.mp4": "Truthful",
    "can_you_tell_when_someone_is_lying_to_you_3_truth.mp4": "Truthful",
    "can_you_tell_when_someone_is_lying_to_you_4_lie.mp4": "Deceptive",
    "can_you_tell_when_someone_is_lying_to_you_5_truth.mp4": "Truthful",
    "can_you_tell_when_someone_is_lying_to_you_6_true.mp4": "Truthful",
}

FEATURE_NAME_TO_PATH = {
    "audio_data": str(AUDIO_FEATURES_DIR),
    "gaze_data": str(GAZE_FEATURES_DIR),
    "micro_expression_data": str(MICRO_EXPRESSION_FEATURES_DIR),
}


def extract_youtube_data_features():
    video_list = add_youtube_data()
    # micro expressions features:
    dict_input_output, output_filename_list = extract_open_face_features(
        video_list,
        path_dir=str(MICRO_EXPRESSION_FEATURES_DIR),
        prefix_name="micro_expressions_youtube_",
        mode=["pose", "aus"],
    )
    annotate_openface_output(dict_input_output, str(MICRO_EXPRESSION_FEATURES_DIR))
    re_annotate_openface_output(str(MICRO_EXPRESSION_FEATURES_DIR))
    # gaze features :
    dict_input_output, output_filename_list = extract_open_face_features(
        video_list,
        path_dir=str(GAZE_FEATURES_DIR),
        prefix_name="gaze_youtube_",
        mode="gaze",
    )
    annotate_openface_output(dict_input_output, str(GAZE_FEATURES_DIR))
    re_annotate_openface_output(str(GAZE_FEATURES_DIR))
    # audio_features
    dict_input_output, output_filename_list = convert_to_mp4_to_wav_files(
        video_list, output_dir=str(AUDIO_FEATURES_DIR), prefix_name="audio_youtube"
    )
    count_number_of_wav_files(str(AUDIO_FEATURES_DIR))
    annotate_audio_files(dict_input_output, dir_path=str(AUDIO_FEATURES_DIR))
    extract_audio_features(input_dir=str(AUDIO_FEATURES_DIR), output_dir=str(AUDIO_FEATURES_DIR))
    convert_arrf_file_to_csv(str(AUDIO_FEATURES_DIR))
    re_annotate_audio_data_and_combine_csvs(str(AUDIO_FEATURES_DIR))


def prepare_youtube_data_for_assessment():
    count_number_of_files_per_model(FEATURE_NAME_TO_PATH)
    show_data(FEATURE_NAME_TO_PATH)
    audio_data, gaze_data, micro_expressions_data = remove_feature_name_annotation(FEATURE_NAME_TO_PATH)
    filename_to_label, filenames = check_and_prepare_labels(audio_data, gaze_data, micro_expressions_data)
    filename_to_encoded_label = encode_labels(filename_to_label)
    df_micro_expressions, df_audio, df_gaze, df_label = clean_and_prepare_data(
        audio_data, gaze_data, micro_expressions_data, filename_to_encoded_label,filenames
    )
    return df_micro_expressions, df_audio, df_gaze, df_label


def assess_model_with_youtube_data(df_micro_expressions, df_audio, df_gaze, df_label):
    # Load the saved model
    model_a = joblib.load(f"{MODELS_DIR}/audio_model.pkl")
    model_m = joblib.load(f"{MODELS_DIR}/micro_expressions_model.pkl")
    model_g = joblib.load(f"{MODELS_DIR}/gaze_model.pkl")

    y_pred_a = model_a.predict(df_audio)
    y_pred_m = model_m.predict(df_micro_expressions)
    y_pred_g = model_g.predict(df_gaze)

    # Combine the predictions using majority voting
    y_pred = mode([y_pred_a, y_pred_m, y_pred_g], axis=0, keepdims=True)[0][0]

    # Calculate the accuracy of the combined predictions
    print(y_pred)
    print(df_label)
    accuracy = accuracy_score(df_label, y_pred)
    print("combined")
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


if __name__ == "__main__":
    # extract_youtube_data_features()
    df_micro_expressions, df_audio, df_gaze, df_label = prepare_youtube_data_for_assessment()
    assess_model_with_youtube_data(df_micro_expressions, df_audio, df_gaze, df_label)
