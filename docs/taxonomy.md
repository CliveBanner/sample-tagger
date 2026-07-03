# Instrument taxonomy (agreed 2026-07-02)

18 classes. Ground rules: every class must be decidable by ear in ~5 seconds; when
timbre is ambiguous, the tiebreak is **role/envelope, not timbre**; if genuinely
undecidable, **skip** (especially in gold mode — a wrong eval label costs more than a
missing one).

## Classes & decision rules

| Class | Rule |
|---|---|
| `kick` | single kick hit (incl. 808 kick one-shots used as kicks) |
| `snare_clap` | single snare, rim, or clap hit |
| `hats_cymbals` | hat/cymbal hits **and** hats-only top loops (one element, even if looped) |
| `tom` | single tom hit |
| `perc` | non-kit percussion: congas, shakers, tambourine, woodblocks — loop or one-shot |
| `drums` | **multiple kit elements together**: breakbeats, full grooves, kick+top loops |
| `bass` | fundamental below ~200 Hz carries the sound; incl. 808/sub *musical* lines |
| `piano_keys` | struck keys with natural decay: piano, EP, rhodes, clav |
| `organ` | sustained drawbar/organ tone (no decay while held) |
| `mallet` | struck-resonant: vibes, marimba, bells, chimes — only if clearly struck-resonant, else → synth |
| `guitar` | recognizable guitar (acoustic/electric), incl. plucked loops |
| `strings` | bowed strings, string sections (real or clearly imitative) |
| `brass` | trumpets, trombones, horns, brass stabs |
| `winds` | flutes, saxes, clarinets, ethnic winds |
| `synth` | synthetic leads, plucks, arps, stabs — and the fallback for "has notes, source unclear" |
| `pad` | sustained, soft/no transient, atmospheric — decided by envelope, not timbre |
| `vocal` | voice: phrases, chops, adlibs, choirs, spoken |
| `sfx` | foley, risers, impacts, noise, field recordings — fallback for "no notes, no beat" |

## Crossover sounds (multi-label)

The label system is **multi-label** (since 2026-07-02): a sample carries a SET of labels
(`sample_labels` table; rank 1 = dominant, drives the `human_instrument` projection used
by map/stats/queues). In the review UI: shift+click (or shift+enter) toggles extra labels,
then click the dominant one — e.g. a synth-processed guitar = `guitar + synth`.

The model is one-vs-rest: 18 independent binary heads, each answering "does this contain
X?". Extra labels are real training positives, predictions above the confidence threshold
land in `model_labels` (top-1 also fills `model_instrument`), and eval is per-class binary
F1 on the gold set's label sets.

Use the ladder below only when no single character dominates *and* no combination fits.

**Tagging discipline:** absent labels are trained as negatives ("no guitar head fires on
this file"). So when a second class *clearly* applies, add it — leaving it off actively
teaches the model that the sound is NOT that class. Don't agonize over borderline
secondaries though: low-confidence maybes are better skipped than guessed, same as
primaries.

## Tiebreak ladder (ambiguous file)

1. Multiple kit elements? → `drums`
2. Recognizable acoustic source? → that class
3. Has pitched notes? → `pad` if sustained/atmospheric, else `synth`
4. Rhythmic but not kit? → `perc`
5. None of the above → `sfx`
6. Still torn → **skip**

## Migration from the previous 17-class set

Pure addition (`drums`); no renames, no splits — all existing human labels remain valid.
Full drum loops previously labeled as an element (kick/perc) are wrong under the new rule;
the Phase-2 confident-learning pass should surface most of them.

Weak-map change: `drums → drums` (was `drums → perc`).

Gold-set note: candidates were stratified over the old model classes, so `drums` gets no
dedicated slice until the first retrain with drums labels; then top up via the gold panel
("＋ Add candidates") to give it val support.
