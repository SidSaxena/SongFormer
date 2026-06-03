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
