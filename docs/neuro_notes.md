# Neuro notes — why the modules are called what they are called

Hakisa 1.0 doubled as a neurology notebook (Thalamus, Limbic System, memory consolidation, H.M.).
Hakisa 2.0 keeps the habit. These are *analogies* for orientation, not claims that the code
models the brain; references point to standard sources.

| Module | Brain structure | The (loose) analogy |
|---|---|---|
| `thalamus` | Thalamus | Relay of almost all sensory afferences to the cortex and of motor loops back out — here: screen in, keys/mouse out, plus the recorder of the human's inputs. |
| `cortex` | Visual cortex (V1 → V2/V4, MT/V5) | Early layers respond to local edges/blobs (Hubel & Wiesel), later areas to patterns and objects; MT/V5 is specialised in motion — Hakisa gets motion from stacked frames. |
| `basal_ganglia` | Basal ganglia | Classic model of *action selection*: candidate motor programs compete, the winner is released from tonic inhibition; dopamine-signalled reward prediction errors tune the selection (Schultz, Dayan & Montague, *Science* 1997) — the critic's TD error is the textbook computational counterpart. |
| pointer head | Superior colliculus | Holds a topographic motor map: the location of activity on the map encodes the next saccade's target (Gandhi & Katnani, *Annu. Rev. Neurosci.* 2011). Hakisa's pointer is a map over the screen whose peak is where the mouse goes. |
| fovea head (JigokuNet) | Fovea / attention | Only a small part of the visual field is seen sharply; gaze is directed where the information is. JigokuNet learns where to "look" (around the ship). |
| danger head | Amygdala | Threat detection and fast fear responses — here: "will I lose a life within one second?" |
| `limbic` | Limbic system / nucleus accumbens / VTA | Reward and motivation (dopamine from the VTA onto the nucleus accumbens, as in Hakisa 1.0's notes) — here: rewards read from the HUD / game client. |
| `hippocampus` | Hippocampus | Episodic memory and *replay* of recent experience during rest (Wilson & McNaughton, *Science* 1994), the inspiration usually cited for experience replay in deep RL. |
| `study` → `consolidation` | Memory consolidation | First learn by imitation (supervised), then stabilise and improve with experience (RL) — Hakisa 1.0's two phases, kept. |
| `prefrontal` | Prefrontal cortex | Executive functions: planning, working memory, following verbal rules, inhibiting impulsive responses. The LLM plans slowly while the reflexes (System 1) act fast — Kahneman's *Thinking, Fast and Slow* terminology, also used by Helix and Hi Robot. |
| HG-DAgger takeover | Motor learning from correction | Errors are most informative exactly where the learner fails — the cerebellar "error signal" story (Marr–Albus–Ito) in a very loose sense. |
