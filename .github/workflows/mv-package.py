import os
import shutil
import glob
import sys
import requests
from datetime import datetime

vai_rt_build_dir = os.environ["VAI_RT_BUILD_DIR"]
files = glob.glob(os.path.join(vai_rt_build_dir, "voe-win_amd64-with_xcompiler*.zip"))
src_file = files[0]

src_name = os.path.splitext(os.path.basename(src_file))[0]
target_name = f"pr{sys.argv[1]}.{src_name}.zip"
upload_file = os.path.join(vai_rt_build_dir, target_name)
shutil.move(src_file, upload_file)
url = "http://xcdl220229:8081"
post_url = f"{url}/upload"
download_url = f"{url}/download/{target_name}"
files = {"file": open(upload_file, "rb")}
response = requests.post(post_url, files=files)

if response.status_code == 200 or response.status_code == 204:
    print(f"package in {download_url}")
    print("POST request succeeded")
    print(f"::set-output name=download_url::{download_url}")
    sys.exit(0)
else:
    print("POST request failed")
    print(f"::set-output name=download_url::{download_url}")
    sys.exit(1)
