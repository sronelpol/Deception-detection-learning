# Deception detection project

#### The project was build using windows as OpenFace wasn't working properly on Mac with Arm chip

## Datasets taken from:

- YouTube's data taken from the following video: [link](https://www.youtube.com/watch?v=Jfli-6Q-13Q&t=21s)

- Real trial testimonies taken
  from:[link](http://web.eecs.umich.edu/~mihalcea/downloads/RealLifeDeceptionDetection.2016.zip)

## pre requisites:

1. install OpenFace outside this folder following those
   instructions: [link](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation)
   - after extracting the folder, run `download_models.ps1` using powershell

2. download OpenSmile outside this folder(download and
   extract): [link](https://github.com/audeering/opensmile/releases/download/v3.0.1/opensmile-3.0.1-win-x64.zip)
3. download ffmpeg : [link](https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z)
   - use the following [link](https://bobbyhadz.com/blog/ffmpeg-is-not-recognized-as-internal-or-external-command) to understand how to add fmpeg to path
4. use python 3.8 or higher (code tested on 3.8.16)
5. install python requirements by typing ```pip install -r ./requirements.txt ```


## Run the code:
1. Go to test_your_video folder and run the gui.py file. You will see the following window:
   ![image](https://user-images.githubusercontent.com/73016570/230710749-05c70c5b-fe10-4bdc-bbd0-8088ac5cd12e.png)
2. Select your own video:
   ![image](https://user-images.githubusercontent.com/73016570/230711000-a9095871-5b73-4adc-945c-7cf458678763.png)

3. see the process running

4. Check ouy the result

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
