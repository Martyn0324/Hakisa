"""Prefrontal cortex — System 2: a language model (Qwen3) that plans and steers System 1.

The prefrontal cortex is associated with executive functions: planning, working memory, inhibition of
impulsive actions, and following goals/rules given verbally — exactly the role this module plays for
Hakisa: slow, deliberate, language-driven plans ("save the bombs for the boss", "go bot"), while the
fast reflexes stay in System 1 (Kahneman's "fast and slow" terminology, used by Helix / Hi Robot).
"""

from hakisa.prefrontal.intents import IntentSchema
from hakisa.prefrontal.reasoner import Prefrontal, make_completion, make_prompt, s1_visual_tokens

__all__ = ["IntentSchema", "Prefrontal", "make_completion", "make_prompt", "s1_visual_tokens"]
