import os
import time
import requests
from huggingface_hub import snapshot_download
def download_model():
    ProjectDir = os.path.abspath(os.path.dirname(__file__))
    CheckpointsDir = os.path.join(ProjectDir, "models")

    def print_directory_contents(path):
        for child in os.listdir(path):
            child_path = os.path.join(path, child)
            if os.path.isdir(child_path):
                print(child_path)

    #if not os.path.exists(CheckpointsDir):
    if True:
        os.makedirs(CheckpointsDir)
        print("Checkpoint Not Downloaded, start downloading...")
        tic = time.time()
        snapshot_download(
            repo_id="TMElyralab/MuseTalk",
            local_dir=CheckpointsDir,
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True, resume_download=False
        )
        # weight
        os.makedirs(f"{CheckpointsDir}/sd-vae-ft-mse/")
        snapshot_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            local_dir=CheckpointsDir+'/sd-vae-ft-mse',
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True, resume_download=False
        )
        #dwpose
        os.makedirs(f"{CheckpointsDir}/dwpose/")
        snapshot_download(
            repo_id="yzd-v/DWPose",
            local_dir=CheckpointsDir+'/dwpose',
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True, resume_download=False
        )
        #vae
        #url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
        url = "https://huggingface.co/cocktailpeanut/klatesum/resolve/main/tiny.pt?download=true"
        response = requests.get(url)
        # 确保请求成功
        if response.status_code == 200:
            # 指定文件保存的位置
            file_path = f"{CheckpointsDir}/whisper/tiny.pt"
            os.makedirs(f"{CheckpointsDir}/whisper/")
            # 将文件内容写入指定位置
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"请求失败，状态码：{response.status_code}")

        # face parse
        url = "https://huggingface.co/cocktailpeanut/klatesum/resolve/main/79999_iter.pth?download=true"
        response = requests.get(url)
        if response.status_code == 200:
            file_path = f"{CheckpointsDir}/face-parse-bisent/79999_iter.pth"
            os.makedirs(f"{CheckpointsDir}/face-parse-bisent/")
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"请求失败，状态码：{response.status_code}")

        #resnet
        url = "https://huggingface.co/cocktailpeanut/klatesum/resolve/main/resnet18-5c106cde.pth?download=true"
        response = requests.get(url)
        if response.status_code == 200:
            file_path = f"{CheckpointsDir}/face-parse-bisent/resnet18-5c106cde.pth"
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"请求失败，状态码：{response.status_code}")


        toc = time.time()

        print(f"download cost {toc-tic} seconds")
        print_directory_contents(CheckpointsDir)

    else:
        print("Already download the model.")





download_model()  # for huggingface deployment.
