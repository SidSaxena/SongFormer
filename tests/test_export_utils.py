import json
import os
import zipfile

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

import export_utils


def test_format_time():
    assert export_utils.format_time(0.0) == "00:00.000"
    assert export_utils.format_time(12.34) == "00:12.340"
    assert export_utils.format_time(61.5) == "01:01.500"


def test_stem_of():
    assert export_utils.stem_of("/a/b/Song.mp3") == "Song"
    assert export_utils.stem_of("Track.wav") == "Track"


def test_segments_to_csv_basic():
    segments = [
        {"start": "0.0", "end": "12.34", "label": "intro"},
        {"start": "12.34", "end": "45.67", "label": "verse"},
    ]
    csv_text = export_utils.segments_to_csv(segments)
    lines = csv_text.strip().split("\n")
    assert lines[0] == "start_sec,start_mmss,end_sec,end_mmss,label"
    assert lines[1] == "0.00,00:00.000,12.34,00:12.340,intro"
    assert lines[2] == "12.34,00:12.340,45.67,00:45.670,verse"
    assert len(lines) == 3


def test_segments_to_csv_quotes_label_with_comma():
    segments = [{"start": "0.0", "end": "1.0", "label": "a,b"}]
    csv_text = export_utils.segments_to_csv(segments)
    assert '"a,b"' in csv_text
