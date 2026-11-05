"""Hakisa 2.0 — a game-playing agent that sees the screen and presses the keys.

Module map (names kept from Hakisa 1.0's neuroscience notebook):

* ``thalamus``      sensory/motor relay: screen capture, input injection, input recording
* ``hippocampus``   memory: episode storage and datasets (experience replay for the cortex)
* ``cortex``        visual encoder trained from scratch
* ``basal_ganglia`` action selection heads (incl. the superior-colliculus pointer head)
* ``limbic``        reward system: HUD probes, reward functions, LoL live client
* ``prefrontal``    System 2: Qwen3 planner that sends intents to the fast policy
* ``study``         supervised learning (behaviour cloning) on human gameplay
* ``consolidation`` reinforcement-learning fine-tuning (PPO anchored to the studied policy)
"""

__version__ = "2.0.0"
