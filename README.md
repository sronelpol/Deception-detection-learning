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
1. Go to test_your_video folder and run the gui.py file. You will see the following window:\
![image](https://user-images.githubusercontent.com/73016570/230714630-ee8e7452-84b3-41e2-bf01-4f388df4aac8.png)
2. Select your own video:\
   ![image](https://user-images.githubusercontent.com/73016570/230714684-9ec6ee5c-c0e6-49f5-bdf6-38d6ca54819b.png)

3. see the process running\
![image](https://user-images.githubusercontent.com/73016570/230714691-a29b7b6d-1d7d-4aad-b8e2-4eeeca49b70f.png)

4. Check out the result:\
   ![image](https://user-images.githubusercontent.com/73016570/230714572-08307b56-c094-4a21-845a-ab1f7bee9bdd.png)


[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
