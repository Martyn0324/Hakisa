import pytest

from hakisa.actions import ActionSpace, InputTimeline, tracked_buttons, tracked_keys
from hakisa.config import available_games, load_game_config
from hakisa.types import InputEvent, InputSnapshot, Region


def snap(t, keys=(), buttons=(), cursor=(0, 0)):
    return InputSnapshot(t, frozenset(keys), frozenset(buttons), cursor)


def test_load_all_game_configs():
    games = available_games()
    assert {"jigoku", "bullet_heaven2", "lol", "dojo_keyboard", "dojo_mouse"} <= set(games)
    for g in games:
        cfg = load_game_config(g)
        space = cfg.action_space
        names = [h.name for h in space.heads]
        assert len(names) == len(set(names))
        assert space.label_keys()


def test_overrides():
    cfg = load_game_config("jigoku", overrides=["obs.height=96", "training.bc.epochs=3"])
    assert cfg.obs.height == 96
    assert cfg.section("training.bc.epochs") == 3


def test_move_label_from_wasd():
    space = load_game_config("jigoku").action_space
    move = space.head("move")
    region = Region(0, 0, 100, 100)
    tl = InputTimeline([snap(0.0, keys={"w", "a"})], [])
    assert move.choices[space.labels_from_timeline(tl, 0.0, region)["move"]].name == "up_left"
    tl = InputTimeline([snap(0.0, keys={"w"})], [])
    assert move.choices[space.labels_from_timeline(tl, 0.0, region)["move"]].name == "up"
    tl = InputTimeline([snap(0.0)], [])
    assert move.choices[space.labels_from_timeline(tl, 0.0, region)["move"]].name == "none"


def test_lookahead_uses_future_state():
    space = load_game_config("jigoku").action_space  # lookahead 100 ms
    region = Region(0, 0, 100, 100)
    events = [InputEvent(0.08, "key", "d", True)]
    tl = InputTimeline([snap(0.0)], events)
    lab = space.labels_from_timeline(tl, 0.0, region)
    assert space.head("move").choices[lab["move"]].name == "right"


def test_tap_label_edge_only():
    space = load_game_config("jigoku").action_space
    region = Region(0, 0, 100, 100)
    # X held the whole time but no press event in the window -> no bomb
    tl = InputTimeline([snap(0.0, keys={"x"}), snap(0.05, keys={"x"})], [])
    assert space.labels_from_timeline(tl, 0.05, region)["bomb"] == 0
    # press event inside [t+lookahead, t+lookahead+tick) -> bomb
    tl = InputTimeline([snap(0.0)], [InputEvent(0.12, "key", "x", True)])
    assert space.labels_from_timeline(tl, 0.0, region)["bomb"] == 1
    # press event outside the window -> no bomb
    tl = InputTimeline([snap(0.0)], [InputEvent(0.30, "key", "x", True)])
    assert space.labels_from_timeline(tl, 0.0, region)["bomb"] == 0


def test_pointer_roundtrip():
    space = load_game_config("bullet_heaven2").action_space
    p = space.head("pointer")
    region = Region(100, 50, 800, 600)
    fh, fw = p.fine_hw
    assert (fh, fw) == (144, 192)
    for x, y in [(100, 50), (500, 350), (899, 649), (437, 123)]:
        fx, fy = space.pixel_to_pointer(p, x, y, region)
        assert 0 <= fx < fw and 0 <= fy < fh
        px, py = space.pointer_to_pixel(p, fx, fy, region)
        assert abs(px - x) <= region.width / fw and abs(py - y) <= region.height / fh


def test_bullet_heaven_labels_and_desired():
    space = load_game_config("bullet_heaven2").action_space
    region = Region(0, 0, 192, 144)
    tl = InputTimeline(
        [snap(0.0, buttons={"left"}, cursor=(50, 60))],
        [InputEvent(0.11, "button", "right", True, 50, 60)],
    )
    lab = space.labels_from_timeline(tl, 0.0, region)
    assert lab["fire"] == 1 and lab["bomb"] == 1 and lab["slow"] == 0
    assert (lab["pointer_x"], lab["pointer_y"]) == (50, 60)
    desired = space.to_desired(lab, region)
    assert desired.buttons == frozenset({"left"})
    assert desired.cursor == (50, 60)
    assert len(desired.taps) == 1 and desired.taps[0].button == "right"


def test_lol_priority_and_conditioned_pointer():
    cfg = load_game_config("lol")
    space = cfg.action_space
    region = Region(0, 0, 1920, 1080)
    minimap = Region(1645, 805, 270, 270)
    at = space.head("action_type")
    # Q press + right click in the same window -> q wins (higher priority); pointer from the Q event
    events = [
        InputEvent(0.06, "button", "right", True, 300, 300),
        InputEvent(0.07, "key", "q", True, 1000, 500),
    ]
    tl = InputTimeline([snap(0.0, cursor=(0, 0))], events)
    lab = space.labels_from_timeline(tl, 0.0, region, minimap)
    assert at.choices[lab["action_type"]].name == "q"
    px, py = space.pointer_to_pixel("pointer", lab["pointer_x"], lab["pointer_y"], region)
    assert abs(px - 1000) <= 8 and abs(py - 500) <= 8
    assert lab["minimap_pointer_x"] == -1  # inactive head
    # ctrl+q -> level_q (more specific than q)
    tl = InputTimeline([snap(0.0, keys={"ctrl"})], [InputEvent(0.06, "key", "q", True)])
    assert at.choices[space.labels_from_timeline(tl, 0.0, region, minimap)["action_type"]].name == "level_q"
    # right click inside the minimap -> minimap_move with the minimap pointer active
    tl = InputTimeline([snap(0.0)], [InputEvent(0.06, "button", "right", True, 1700, 900)])
    lab = space.labels_from_timeline(tl, 0.0, region, minimap)
    assert at.choices[lab["action_type"]].name == "minimap_move"
    assert lab["minimap_pointer_x"] >= 0 and lab["pointer_x"] == -1
    # holding the right button (dragging) with no new click -> repeated move_click
    tl = InputTimeline([snap(0.0, buttons={"right"}, cursor=(400, 400))], [])
    assert at.choices[space.labels_from_timeline(tl, 0.0, region, minimap)["action_type"]].name == "move_click"


def test_lol_to_desired_quick_cast():
    space = load_game_config("lol").action_space
    region = Region(0, 0, 1920, 1080)
    minimap = Region(1645, 805, 270, 270)
    act = space.noop_action()
    act["action_type"] = space.head("action_type").index("e")
    fx, fy = space.pixel_to_pointer("pointer", 960, 540, region)
    act["pointer_x"], act["pointer_y"] = fx, fy
    d = space.to_desired(act, region, minimap)
    assert d.keys == frozenset() and len(d.taps) == 1
    tap = d.taps[0]
    assert tap.keys == ("e",) and abs(tap.cursor[0] - 960) <= 8 and abs(tap.cursor[1] - 540) <= 8
    act["action_type"] = space.head("action_type").index("level_r")
    assert space.to_desired(act, region, minimap).taps[0].keys == ("ctrl", "r")


def test_to_desired_jigoku():
    space = load_game_config("jigoku").action_space
    act = {"move": space.head("move").index("down_right"), "focus": 1, "shoot": 1, "bomb": 1}
    d = space.to_desired(act, Region(0, 0, 10, 10))
    assert d.keys == frozenset({"s", "d", "shift", "z"})
    assert [t.keys for t in d.taps] == [("x",)]
    assert d.cursor is None


def test_flip_permutation():
    space = load_game_config("jigoku").action_space
    move = space.head("move")
    perm = space.flip_permutation(move)
    names = [c.name for c in move.choices]
    assert names[perm[names.index("up_left")]] == "up_right"
    assert names[perm[names.index("down")]] == "down"


def test_tracked_inputs():
    space = load_game_config("bullet_heaven2").action_space
    assert tracked_keys(space) == ["shift"]
    assert set(tracked_buttons(space)) == {"left", "right"}


def test_unknown_pointer_rejected():
    from hakisa.actions import Choice, HeadSpec

    with pytest.raises(ValueError):
        ActionSpace([HeadSpec("a", "categorical", choices=(Choice("x", keys=("q",), pointer="nope"),))])
