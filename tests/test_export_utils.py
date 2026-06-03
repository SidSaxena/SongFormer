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


def test_write_exports(tmp_path):
    segments = [{"start": "0.0", "end": "1.0", "label": "intro"}]
    json_str = json.dumps(segments)
    msa_str = "0.00 intro\n1.00 end"
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])

    paths = export_utils.write_exports(
        "/some/dir/Song.mp3", segments, json_str, msa_str, fig, str(tmp_path)
    )

    assert os.path.basename(paths["json"]) == "Song.json"
    assert os.path.basename(paths["msa"]) == "Song.msa.txt"
    assert os.path.basename(paths["csv"]) == "Song.csv"
    assert os.path.basename(paths["png"]) == "Song.png"
    # Every file exists and lives in tmp_path
    for p in paths.values():
        assert os.path.isfile(p)
    # Contents round-trip
    with open(paths["json"], encoding="utf-8") as f:
        assert json.load(f) == segments
    with open(paths["msa"], encoding="utf-8") as f:
        assert f.read() == msa_str
    assert os.path.getsize(paths["png"]) > 0


def test_make_zip(tmp_path):
    f1 = tmp_path / "a.json"
    f1.write_text("{}")
    f2 = tmp_path / "a.csv"
    f2.write_text("x")
    zip_path = str(tmp_path / "a_songformer.zip")

    returned = export_utils.make_zip([str(f1), str(f2)], zip_path)

    assert returned == zip_path
    with zipfile.ZipFile(zip_path) as zf:
        assert sorted(zf.namelist()) == ["a.csv", "a.json"]


def test_cleanup_old_exports(tmp_path):
    parent = tmp_path / "exports"
    parent.mkdir()
    old = parent / "run_old"
    old.mkdir()
    (old / "f.txt").write_text("x")
    new = parent / "run_new"
    new.mkdir()
    (new / "f.txt").write_text("x")
    now = 1_000_000.0
    os.utime(old, (now - 7200, now - 7200))  # 2h old
    os.utime(new, (now - 600, now - 600))  # 10m old

    removed = export_utils.cleanup_old_exports(str(parent), 3600, now=now)

    assert [os.path.basename(p) for p in removed] == ["run_old"]
    assert not old.exists()
    assert new.exists()


def test_cleanup_old_exports_missing_parent(tmp_path):
    missing = str(tmp_path / "does_not_exist")
    assert export_utils.cleanup_old_exports(missing, 3600) == []


def test_cleanup_old_exports_ignores_files(tmp_path):
    parent = tmp_path / "exports"
    parent.mkdir()
    stray = parent / "stray.txt"
    stray.write_text("x")
    now = 1_000_000.0
    os.utime(stray, (now - 7200, now - 7200))

    removed = export_utils.cleanup_old_exports(str(parent), 3600, now=now)

    assert removed == []
    assert stray.exists()  # only directories are swept


def test_segments_to_table():
    segments = [
        {"start": "0.0", "end": "12.34", "label": "intro"},
        {"start": "12.34", "end": "45.67", "label": "verse"},
    ]
    rows = export_utils.segments_to_table(segments)
    assert rows == [
        ["0.00 (00:00.000)", "12.34 (00:12.340)", "intro"],
        ["12.34 (00:12.340)", "45.67 (00:45.670)", "verse"],
    ]


def test_segments_to_combined_csv():
    named = [
        ("Song1", [{"start": "0.0", "end": "1.0", "label": "intro"}]),
        ("Song2", [{"start": "0.0", "end": "2.5", "label": "a,b"}]),
    ]
    csv_text = export_utils.segments_to_combined_csv(named)
    lines = csv_text.strip().split("\n")
    assert lines[0] == "filename,start_sec,start_mmss,end_sec,end_mmss,label"
    assert lines[1] == "Song1,0.00,00:00.000,1.00,00:01.000,intro"
    assert lines[2].startswith("Song2,0.00,00:00.000,2.50,00:02.500,")
    assert '"a,b"' in csv_text
    assert len(lines) == 3
