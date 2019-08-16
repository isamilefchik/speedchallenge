import os
import shutil
from os import path

def main():
    for i in range(1120, 1601):
        cur_path = "../data/better_train_frames/t" + str(i) + ".jpg"
        if path.exists(cur_path):
            os.rename(cur_path, "../data/better_train_frames/" + str(19280+i) + ".jpg")

if __name__ == "__main__":
    main()
