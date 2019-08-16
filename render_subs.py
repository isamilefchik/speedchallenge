#!/usr/local/bin/python3
import srt
import argparse
from datetime import timedelta

def main():
    """Main routine.

    This function allows for an .srt file to be created from the model prediction
    text file. The predictions can then be viewed in sync with the dashcam footage
    by setting the .srt as the subtitle file of the video.
    """

    parser = argparse.ArgumentParser(description="Subtitle Renderer")
    parser.add_argument("-i", "--input", type=str, default="", help="Path to speed predictions text.")
    parser.add_argument("-o", "--output", type=str, default="", help="Output file name.")

    args = parser.parse_args()
    filepath, output_path = args.input, args.output

    with open(filepath) as file:
        raw = file.read()
        result = raw.split("\n")

    subtitles = []
    td = timedelta(microseconds=50000)
    cur_time = timedelta(microseconds=0)
    for i, line in enumerate(result):
        subtitles.append(srt.Subtitle(i, cur_time, cur_time + td, line))
        cur_time = cur_time + td

    with open(output_path, 'w') as f:
        f.write(srt.compose(subtitles))

if __name__ == "__main__":
    main()
