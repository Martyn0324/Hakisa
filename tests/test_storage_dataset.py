import numpy as np
import torch

from hakisa.config import load_game_config
from hakisa.hippocampus.dataset import AugmentConfig, BCDataset, ChunkShuffleSampler, compute_class_weights, split_episodes
from hakisa.hippocampus.labeling import compute_labels, danger_array
from hakisa.hippocampus.storage import Episode, EpisodeWriter, list_episodes, load_episodes
from hakisa.types import InputEvent


def _write_human_episode(tmp_path, cfg, n=10, compress=False, chunk=4):
    keys = ["w", "a", "s", "d", "shift", "z", "x"]
    meta = {
        "game": cfg.name,
        "tracked_keys": keys,
        "tracked_buttons": [],
        "obs_region": [0, 0, cfg.obs.width, cfg.obs.height],
        "source": "human",
    }
    w = EpisodeWriter(tmp_path, meta, chunk_size=chunk, compress=compress)
    for i in range(n):
        frame = np.full((cfg.obs.height, cfg.obs.width, 3), i, dtype=np.uint8)
        kmask = np.zeros(len(keys), dtype=bool)
        if i >= 5:
            kmask[keys.index("d")] = True  # holds right from tick 5
        w.add(frame, {"t": i * 0.05, "keys": kmask, "buttons": np.zeros(0, bool), "cursor": np.array([0, 0]),
                      "hud/lives": 3 - (i >= 8)})
    w.add_events([InputEvent(0.25, "key", "d", True), InputEvent(0.31, "key", "x", True)])
    return w.close()


def test_storage_roundtrip(tmp_path):
    cfg = load_game_config("jigoku", overrides=["obs.height=8", "obs.width=6"])
    for compress in (False, True):
        path = _write_human_episode(tmp_path / str(compress), cfg, compress=compress)
        ep = Episode(path)
        assert ep.length == 10 and ep.meta["complete"]
        assert ep.frames(7)[0, 0, 0] == 7
        assert list(ep.frames(np.array([0, 4, 9]))[:, 0, 0, 0]) == [0, 4, 9]
        assert ep.array("hud/lives").tolist() == [3] * 8 + [2] * 2
        assert len(ep.events()) == 2
        assert list_episodes(tmp_path / str(compress)) == [path]


def test_labels_from_recording(tmp_path):
    cfg = load_game_config("jigoku", overrides=["obs.height=8", "obs.width=6", "actions.lookahead_ms=0"])
    ep = Episode(_write_human_episode(tmp_path, cfg))
    lab = compute_labels(ep, cfg.action_space)
    move = cfg.action_space.head("move")
    names = [move.choices[i].name for i in lab["move"]]
    assert names[:5] == ["none"] * 5 and names[5:] == ["right"] * 5
    # x pressed at t=0.31 -> inside tick 6's window [0.30, 0.35)
    assert lab["bomb"].tolist() == [0] * 6 + [1] + [0] * 3


def test_danger_array():
    class Fake:
        length = 6

        def has(self, k):
            return k == "life_lost"

        def array(self, k):
            return np.array([0, 0, 0, 1, 0, 0], dtype=bool)

    d = danger_array(Fake(), horizon=2)
    assert d.tolist() == [0, 1, 1, 0, 0, 0]


def test_dataset_stacks_and_boundaries(tmp_path):
    cfg = load_game_config("jigoku", overrides=["obs.height=8", "obs.width=6", "obs.stack=3", "actions.lookahead_ms=0"])
    paths = [_write_human_episode(tmp_path / f"e{i}", cfg) for i in range(2)]
    eps = load_episodes([p.parent for p in paths])
    for ep in eps:
        np.savez(ep.path / "labels.npz", **compute_labels(ep, cfg.action_space))
    eps = load_episodes([p.parent for p in paths])
    ds = BCDataset(eps, cfg)
    assert len(ds) == 20
    item = ds[0]
    assert item["obs"].shape == (9, 8, 6) and item["obs"].dtype == torch.uint8
    # first tick: all three stacked frames are frame 0 (no crossing into a previous episode)
    assert item["obs"][:, 0, 0].tolist() == [0] * 9
    item = ds[11]  # second episode, t=1 -> frames (0, 0, 1)
    assert item["obs"][::3, 0, 0].tolist() == [0, 0, 1]
    assert set(item["labels"]) == set(cfg.action_space.label_keys())
    weights = compute_class_weights(ds)
    assert weights["bomb"][1] > 1.0 > weights["bomb"][0]
    sampler = ChunkShuffleSampler(ds, group=2)
    assert sorted(list(sampler)) == list(range(len(ds)))
    train, val = split_episodes(eps, 0.5)
    assert len(train) == 1 and len(val) == 1


def test_flip_and_shift_augmentation_keeps_pointer_consistent(tmp_path):
    cfg = load_game_config("dojo_mouse", overrides=["obs.height=16", "obs.width=16", "obs.stack=1"])
    space = cfg.action_space
    meta = {"game": cfg.name, "obs_region": [0, 0, 16, 16]}
    w = EpisodeWriter(tmp_path, meta, chunk_size=8)
    frame = np.zeros((16, 16, 3), np.uint8)
    frame[7, 6] = 255  # a bright pixel at x=6, y=7 (away from the border: shifts are <= 3 px) ...
    p = space.head("pointer")
    w.add(frame, {"label/pointer_x": 6, "label/pointer_y": 7, "label/fire": 1, "label/bomb": 0, "label/slow": 0,
                  "cursor_norm": np.array([0.1, 0.2], np.float32)})
    ep = Episode(w.close())
    assert p.fine_hw == (16, 16)
    ds = BCDataset([ep], cfg, augment=AugmentConfig(shift=3, flip=True, color=0.0))
    rng = np.random.default_rng(0)
    for _ in range(20):
        item = ds.item_numpy(0, rng)
        x, y = item["labels"]["pointer_x"], item["labels"]["pointer_y"]
        # ... and the label must still point at it after flip/shift
        assert item["obs"][0, y, x] == 255
        assert item["obs"].shape == (4, 16, 16)  # 3 RGB + cursor channel


def test_history_dropout_and_keyframe_weights(tmp_path):
    from hakisa.hippocampus.dataset import keyframe_weights

    cfg = load_game_config("jigoku", overrides=["obs.height=8", "obs.width=6", "obs.stack=3", "actions.lookahead_ms=0"])
    path = _write_human_episode(tmp_path, cfg)  # move label switches from none to right at tick 5
    ep = Episode(path)
    np.savez(ep.path / "labels.npz", **compute_labels(ep, cfg.action_space))
    ep = Episode(path)
    ds = BCDataset([ep], cfg, augment=AugmentConfig(shift=0, flip=False, color=0.0, history_dropout=1.0))
    item = ds.item_numpy(4, np.random.default_rng(0))  # frames (2, 3, 4) -> all replaced by the newest (4)
    assert item["obs"][:, 0, 0].tolist() == [4] * 9
    w = keyframe_weights(BCDataset([ep], cfg), boost=3.0)
    assert w.tolist() == [1, 1, 1, 1, 1, 3, 1, 1, 1, 1]  # only the tick where "right" starts
