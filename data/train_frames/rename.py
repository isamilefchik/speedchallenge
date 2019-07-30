import os
import shutil
from os import path

def main():
	for i in range(30399):
		if path.exists("a" + str(i) + ".jpg"):
			os.rename("a" + str(i) + ".jpg", str(i+1) + ".jpg")

if __name__ == "__main__":
    main()
