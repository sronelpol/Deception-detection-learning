import glob
import os
from pathlib import Path

DIR = Path(__file__).parent.parent
DATASETS_DIR = DIR / "datasets"


def add_trial_testimonies_data():
    videos = []
    dir_list = [f"{str(DATASETS_DIR)}/trial_testimonies/Deceptive", f"{str(DATASETS_DIR)}/trial_testimonies/Truthful"]
    count_trial = 0
    for _dir in dir_list:
        for filename in glob.glob(os.path.join(_dir, "*.mp4")):
            videos.append(filename)
            count_trial += 1
    print("number of videos from trial data")
    print(count_trial)
    return videos


def add_youtube_data():
    videos = []
    count = 0
    for filename in glob.glob(os.path.join(f"{str(DATASETS_DIR)}/youtube", "*.mp4")):
        videos.append(filename)
        count += 1
    print("number of videos from youtube data")
    print(count)
    return videos
