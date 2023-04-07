from pathlib import Path

from feature_extraction.helpers import add_trial_testimonies_data
from feature_extraction.utils import extract_open_face_features, annotate_openface_output, re_annotate_openface_output

DIR = Path(__file__).parent.parent
RESOURCES_DIR = DIR / "resources"
GAZE_FEATURES_DIR = RESOURCES_DIR / "gaze_features"

if __name__ == '__main__':
    video_list = add_trial_testimonies_data()
    dict_input_output, output_filename_list = (
        extract_open_face_features(video_list,
                                   path_dir=str(GAZE_FEATURES_DIR),
                                   prefix_name="gaze_real_life_deception_",
                                   mode="gaze")
    )
    annotate_openface_output(dict_input_output, str(GAZE_FEATURES_DIR))
    re_annotate_openface_output(str(GAZE_FEATURES_DIR))
