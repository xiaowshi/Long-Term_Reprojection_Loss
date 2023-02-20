import os
import hashlib
import urllib
import zipfile

video_path = "~/SCARED/SCARED_video"
video_url = "https://drive.google.com/uc?id=1c_ewx6wts7pJTb3XVWVXEZLbFXDObpdP"
print("-> Downloading videos to {}".format(video_path + ".zip"))
urllib.request.urlretrieve(video_url, video_path + ".zip")



print("   Unzipping videos ...")
with zipfile.ZipFile(video_path + ".zip", 'r') as f:
    f.extractall(video_path)

print("   Videos unzipped to {}".format(video_path))
