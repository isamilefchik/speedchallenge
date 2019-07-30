import srt
from datetime import timedelta

filepath = "test_result.txt"
with open(filepath) as file:
    raw = file.read()
    result = list(map(float, raw.split("\n")))

subtitles = []
td = timedelta(microseconds=50000)
cur_time = timedelta(microseconds=0)
for i, speed in enumerate(result):
    subtitles.append(srt.Subtitle(i, cur_time, cur_time + td, str(speed)))
    cur_time = cur_time + td

with open('test_speedsubs.srt', 'w') as f:
    f.write(srt.compose(subtitles))