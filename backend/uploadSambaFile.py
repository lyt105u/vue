# usage: python uploadSambaFile.py a2556 rkothebest51 "\\LAPTOP-BENDAENK\tmp\高醫訓練csv.csv" upload
import smbclient
import shutil
import argparse
import json
import os

def main(args):
    try:
        smbclient.ClientConfig(username=args.username, password=args.password)

        filename = os.path.basename(args.remote_path)
        local_full_path = os.path.join(args.folder, filename)

        with smbclient.open_file(args.remote_path, mode="rb") as remote_file:
            with open(local_full_path , mode="wb") as local_file:
                shutil.copyfileobj(remote_file, local_file)

        print(json.dumps({
            "status": "success",
        }))
    
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": f"{e}",
        }))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload file from SMB share to local.")
    parser.add_argument("username", help="SMB username")
    parser.add_argument("password", help="SMB password")
    parser.add_argument("remote_path", help="Remote SMB file path")
    parser.add_argument("folder", help="Save file to this folder")
    args = parser.parse_args()

    main(args)
