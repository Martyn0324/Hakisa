import numpy as np

from hakisa.envs.dojo_render import FONT, draw_text
from hakisa.limbic.probes import (
    DigitTemplateProbe,
    HudReader,
    LocateProbe,
    PixelCountProbe,
    RegionChangeProbe,
    TemplateProbe,
    build_probes,
    load_digit_templates,
)
from hakisa.limbic.rewards import BulletHellReward


def _hud_image(lives=3, score=12345):
    img = np.zeros((60, 200, 3), np.uint8)
    for k in range(lives):
        img[5:15, 5 + 14 * k : 15 + 14 * k] = (255, 64, 64)
    draw_text(img, str(score), 5, 30, (255, 255, 255))
    return img


def test_pixel_count_probe():
    probe = PixelCountProbe("lives", (0, 0, 200, 20), color=(255, 64, 64), tol=20, unit_pixels=100)
    assert probe.read(_hud_image(lives=3)) == 3
    assert probe.read(_hud_image(lives=0)) == 0


def test_digit_probe_reads_numbers():
    probe = DigitTemplateProbe("score", (0, 25, 200, 30), templates=load_digit_templates("builtin:dojo"))
    for value in (0, 7, 12345, 908172):
        assert probe.read(_hud_image(score=value)) == value
    assert probe.read(np.zeros((60, 200, 3), np.uint8)) is None


def test_digit_probe_from_user_templates():
    # user-style templates: RGB crops of each glyph
    templates = {}
    for d in "0123456789":
        g = np.kron(FONT[d], np.ones((3, 3), bool))
        img = np.zeros(g.shape + (3,), np.uint8)
        img[g] = (255, 255, 255)
        templates[d] = img
    probe = DigitTemplateProbe("score", (0, 25, 200, 30), templates=templates)
    assert probe.read(_hud_image(score=4096)) == 4096  # different scale than the templates


def test_template_and_locate_probes():
    rng = np.random.default_rng(0)
    screen = rng.integers(0, 40, (80, 120, 3)).astype(np.uint8)
    sprite = np.zeros((6, 6, 3), np.uint8)
    sprite[1:5, 1:5] = (250, 250, 250)
    sprite[2:4, 2:4] = (20, 200, 20)
    screen[30:36, 50:56] = sprite
    assert TemplateProbe("go", (0, 0, 120, 80), sprite, threshold=0.8).read(screen) == 1
    assert TemplateProbe("go", (0, 0, 40, 30), sprite, threshold=0.8).read(screen) == 0
    loc = LocateProbe("ship", (0, 0, 120, 80), sprite, threshold=0.8).read(screen)
    assert loc == (53, 33)


def test_change_probe():
    p = RegionChangeProbe("c", (0, 0, 4, 4))
    a = np.zeros((4, 4, 3), np.uint8)
    assert p.read(a) == 0.0
    assert p.read(a + 10) > 9


def test_hud_debounce():
    probe = PixelCountProbe("lives", (0, 0, 200, 20), color=(255, 64, 64), tol=20, unit_pixels=100)
    hud = HudReader({"lives": probe}, stable_frames=2)
    assert hud.read(_hud_image(lives=3))["lives"] == 3
    assert hud.read(_hud_image(lives=1))["lives"] == 3  # one flickering frame is ignored
    assert hud.read(_hud_image(lives=3))["lives"] == 3
    hud.read(_hud_image(lives=2))
    assert hud.read(_hud_image(lives=2))["lives"] == 2


def test_build_probes_skips_uncalibrated():
    specs = [
        {"name": "lives", "type": "pixel_count", "region": [0, 0, 1, 1], "color": [1, 2, 3], "calibrated": False},
        {"name": "score", "type": "digits", "region": [0, 0, 5, 5], "templates": "builtin:dojo"},
    ]
    assert set(build_probes(specs)) == {"score"}
    try:
        build_probes(specs, require_calibrated=True)
        raise AssertionError("expected an error")
    except RuntimeError as exc:
        assert "calibrate" in str(exc)


def test_bullet_hell_reward_signs():
    rf = BulletHellReward({"survival": 0.01, "life_lost": -1.0, "bomb_used": -0.3, "score_scale": 1e-3, "score_clip": 0.5})
    rf.reset({"lives": 3, "bombs": 2, "score": 0})
    r, info = rf({"lives": 3, "bombs": 2, "score": 100})
    assert np.isclose(r, 0.01 + 0.1)
    r, info = rf({"lives": 3, "bombs": 1, "score": 100})
    assert info["bomb_used"] and np.isclose(r, 0.01 - 0.3)
    r, info = rf({"lives": 2, "bombs": 2, "score": 100})  # death resets bombs: not a bomb use
    assert info["life_lost"] and not info["bomb_used"] and np.isclose(r, 0.01 - 1.0)
    r, _ = rf({"lives": 2, "bombs": 2, "score": 100000})
    assert np.isclose(r, 0.01 + 0.5)  # clipped
    r, _ = rf({"lives": 2, "bombs": 2, "score": 5})  # misread (lower score) -> no reward, no reset
    assert np.isclose(r, 0.01)
    r, _ = rf({"lives": 2, "bombs": 2, "score": 100000})
    assert np.isclose(r, 0.01)
    r, info = rf({"lives": None, "bombs": None, "score": None})  # unreadable HUD -> survival only
    assert np.isclose(r, 0.01)
    r, info = rf({"lives": 0, "bombs": 2, "game_over": True})
    assert info["game_over"] and np.isclose(r, 0.01 - 2.0 - 1.0)
