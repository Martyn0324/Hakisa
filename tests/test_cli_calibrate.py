import numpy as np

from hakisa.calibrate import glyphs_from_region, save_digit_templates, unit_pixels
from hakisa.cli import main
from hakisa.envs.dojo_render import draw_text
from hakisa.limbic.probes import DigitTemplateProbe, load_digit_templates


def test_digit_templates_from_a_screenshot(tmp_path):
    img = np.zeros((40, 200, 3), np.uint8)
    draw_text(img, "1234567890", 4, 10, (250, 250, 250), scale=2)
    glyphs = glyphs_from_region(img, (0, 5, 200, 30))
    assert len(glyphs) == 10
    saved = save_digit_templates(img, (0, 5, 200, 30), "1234567890", tmp_path / "digits")
    assert sorted(saved) == list("0123456789")
    probe = DigitTemplateProbe("score", (0, 5, 200, 30), load_digit_templates(str(tmp_path / "digits")))
    other = np.zeros((40, 200, 3), np.uint8)
    draw_text(other, "907", 30, 10, (250, 250, 250), scale=2)
    assert probe.read(other) == 907


def test_unit_pixels():
    img = np.zeros((20, 60, 3), np.uint8)
    for k in range(3):
        img[2:10, 2 + 12 * k : 10 + 12 * k] = (255, 0, 0)
    assert unit_pixels(img, (0, 0, 60, 20), (255, 0, 0), 10, 3) == 64


def test_cli_games_and_dojo_pipeline(tmp_path, capsys):
    main(["games"])
    out = capsys.readouterr().out
    assert "jigoku" in out and "lol" in out
    main(["dojo-demos", "--scheme", "keyboard", "--steps", "256", "--envs", "8", "--out", str(tmp_path / "demos")])
    main(["train-bc", "--game", "dojo_keyboard", "--data", str(tmp_path / "demos"), "--out", str(tmp_path / "bc"),
          "--epochs", "1", "--batch-size", "32", "--workers", "0", "--device", "cpu", "--val-fraction", "0.25"])
    assert (tmp_path / "bc" / "best.pt").exists()
