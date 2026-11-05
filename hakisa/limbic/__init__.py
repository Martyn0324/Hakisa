"""Limbic system: what is "good" for Hakisa (rewards) and how it is read from the screen / game.

"The shining star may probably be the Nucleus Accumbens, which is modulated by dopamine originated
from the Ventral Tegmental Area ... the reward expectancy motivates the subject to make a move."
— Hakisa 1.0 notes.  Here: HUD probes (perception of the outcome), reward functions (the dopamine
signal) and the League of Legends live-client reader.
"""

from hakisa.limbic.probes import HudReader, build_probes
from hakisa.limbic.rewards import BulletHellReward, BulletHellRewardConfig

__all__ = ["BulletHellReward", "BulletHellRewardConfig", "HudReader", "build_probes"]
