"""Hippocampus: episodic memory (recordings on disk) and the datasets the cortex studies from.

"Hippocampus is specialized in work memory and short-term memory, and it's the entrance door for
long-term memory in the brain" — Hakisa 1.0 notes.  Replay of stored experience during learning is
also the classic analogy for experience replay in deep RL (hippocampal replay during sleep:
Wilson & McNaughton, Science 1994).
"""

from hakisa.hippocampus.storage import Episode, EpisodeWriter, list_episodes, load_episodes

__all__ = ["Episode", "EpisodeWriter", "list_episodes", "load_episodes"]
