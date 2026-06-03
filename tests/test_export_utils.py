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
    assert os.path.basename(paths["audacity"]) == "Song.audacity.txt"
    # Every file exists and lives in tmp_path
    for p in paths.values():
        assert os.path.isfile(p)
    # Contents round-trip
    with open(paths["json"], encoding="utf-8") as f:
        assert json.load(f) == segments
    with open(paths["msa"], encoding="utf-8") as f:
        assert f.read() == msa_str
    assert os.path.getsize(paths["png"]) > 0
    with open(paths["audacity"], encoding="utf-8") as f:
        assert f.read() == "0.000000\t1.000000\tintro\n"


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


def test_combined_json():
    named = [
        ("Song1", [{"start": 0.0, "end": 1.0, "label": "intro"}]),
        ("Song2", [{"start": 0.0, "end": 2.0, "label": "verse"}]),
    ]
    data = json.loads(export_utils.combined_json(named))
    assert set(data.keys()) == {"Song1", "Song2"}
    assert data["Song1"] == [{"start": 0.0, "end": 1.0, "label": "intro"}]


def test_zip_dir(tmp_path):
    src = tmp_path / "bundle"
    (src / "Song1").mkdir(parents=True)
    (src / "Song1" / "Song1.json").write_text("{}")
    (src / "summary.csv").write_text("x")
    zip_path = str(tmp_path / "batch.zip")

    returned = export_utils.zip_dir(str(src), zip_path)

    assert returned == zip_path
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
    assert names == ["Song1/Song1.json", "summary.csv"]


def test_write_exports_custom_stem(tmp_path):
    segments = [{"start": "0.0", "end": "1.0", "label": "intro"}]
    fig = Figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])

    paths = export_utils.write_exports(
        "/some/dir/Song.mp3",
        segments,
        "[]",
        "0.00 intro",
        fig,
        str(tmp_path),
        stem="Song_2",
    )

    assert os.path.basename(paths["json"]) == "Song_2.json"
    assert os.path.basename(paths["msa"]) == "Song_2.msa.txt"
    assert os.path.basename(paths["csv"]) == "Song_2.csv"
    assert os.path.basename(paths["png"]) == "Song_2.png"
    assert os.path.basename(paths["audacity"]) == "Song_2.audacity.txt"
    for p in paths.values():
        assert os.path.isfile(p)


def test_zip_dir_stores_precompressed_files(tmp_path):
    src = tmp_path / "bundle"
    src.mkdir()
    (src / "a.png").write_bytes(b"\x89PNG fake image bytes")
    (src / "a.csv").write_text("x,y\n1,2\n")
    zip_path = str(tmp_path / "batch.zip")

    export_utils.zip_dir(str(src), zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        assert zf.getinfo("a.png").compress_type == zipfile.ZIP_STORED
        assert zf.getinfo("a.csv").compress_type == zipfile.ZIP_DEFLATED


def test_new_run_dir(tmp_path):
    parent = tmp_path / "exports"
    # Pre-create a stale run dir that the bootstrap should sweep
    stale = parent / "run_stale"
    stale.mkdir(parents=True)
    old = 1.0  # epoch: definitely older than any TTL
    os.utime(stale, (old, old))

    run_dir = export_utils.new_run_dir(parent_dir=str(parent))

    assert os.path.isdir(run_dir)
    assert os.path.dirname(run_dir) == str(parent)
    assert os.path.basename(run_dir).startswith("run_")
    assert not stale.exists()  # stale run swept by the bootstrap


def test_segments_to_audacity():
    segments = [
        {"start": "0.0", "end": "12.34", "label": "intro"},
        {"start": 12.34, "end": 45.67, "label": "verse"},
    ]
    text = export_utils.segments_to_audacity(segments)
    lines = text.strip().split("\n")
    assert lines[0] == "0.000000\t12.340000\tintro"
    assert lines[1] == "12.340000\t45.670000\tverse"
    assert len(lines) == 2
    assert text.endswith("\n")
