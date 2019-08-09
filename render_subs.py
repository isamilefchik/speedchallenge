import srt
from datetime import timedelta

filepath = "test_set_result.txt"
with open(filepath) as file:
    raw = file.read()
    result = raw.split("\n")

subtitles = []
td = timedelta(microseconds=50000)
cur_time = timedelta(microseconds=0)
for i, line in enumerate(result):
    subtitles.append(srt.Subtitle(i, cur_time, cur_time + td, line))
    cur_time = cur_time + td

with open('test_speedsubs.srt', 'w') as f:
    f.write(srt.compose(subtitles))
