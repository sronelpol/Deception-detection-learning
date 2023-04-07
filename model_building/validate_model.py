from pathlib import Path

DIR = Path(__file__).parent.parent
RESOURCES_DIR = DIR / "resources"
AUDIO_FEATURES_DIR = RESOURCES_DIR / "youtube" / "audio_features" / "csv_full_audio"
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
    'audio_data': str(AUDIO_FEATURES_DIR),
    'gaze_data': str(GAZE_FEATURES_DIR),
    'micro_expression_data': str(MICRO_EXPRESSION_FEATURES_DIR)
}


def prepare_youtube_data():
    data = []
    return data


def validate_model_with_youtube_data():
    pass
