# LLM Cluster Classification Manual

## Goal

Classify audio sample clusters from a DAW sample library (~226k files, ~5,661 clusters) into
instrument categories. You receive a JSON dump of cluster summaries and output a JSON assignment file.

---

## Workflow

### Step 1 — Generate the dump

```bash
python3 scripts/cluster_dump.py --db samples.db \
    --min-n 10 --top 12 --unlabeled-only \
    --output /tmp/cluster_dump.json
```

This outputs a JSON array sorted by cluster size descending (most coverage first).

### Step 2 — Classify (your job, described below)

Read chunks of the dump (e.g. 80–100 clusters at a time). For each cluster, decide on a label or
skip. Output a JSON assignment file.

### Step 3 — Apply

```bash
python3 scripts/cluster_apply.py --db samples.db \
    --input /tmp/cluster_labels_batchN.json \
    --min-confidence med
```

Only writes to samples where `human_instrument IS NULL` — safe to re-run.

---

## Input format (one cluster entry)

```json
{
  "id": 1578,
  "n": 82,
  "n_unlab": 82,
  "agreement": 0.72,
  "model_label": "drums",
  "paths": [
    "ROLAND S 550/SNAP 2A",
    "AMG Gota Yashiki/GOTA3SNARE 1",
    "Back in Time/R SNARE 2"
  ]
}
```

- `id` — cluster ID (primary key in `samples` table)
- `n` — total members in cluster
- `n_unlab` — unlabeled members (what you'd actually label)
- `agreement` — fraction of members sharing the dominant `model_label` (0–1)
- `model_label` — what the audio model predicted for most members
- `paths` — last 2–3 path segments of up to 12 unlabeled members, sorted by cluster distance
  (core members first)

---

## Output format

```json
[
  {
    "id": 1578,
    "label": "snare",
    "confidence": "high",
    "note": "AMG Gota GOTA3SNARE + Back in Time R SNARE — explicit snare; model=drums wrong"
  },
  {
    "id": 1047,
    "label": null,
    "confidence": "low",
    "note": "skip: opaque AMG hits, low agreement 0.47"
  }
]
```

`label`: instrument name from the taxonomy (see below), or `null`/`"skip"` to skip.
`confidence`: `"high"`, `"med"`, or `"low"`. The apply script ignores `low` by default.

---

## Taxonomy (28 labels)

| Label | Covers |
|-------|--------|
| `kick` | Kick drums, bass drums. 808 BD, BD2, KICK, DMX kicks, TX-81Z BD, MBase. |
| `snare` | Snare drums, snare hits. SN, SD, SNARE, snr. Includes clapper/snare variants. |
| `hihat` | Hi-hats (open and closed). HH, HAT, closedhat_, openhat_, 808 HH. |
| `clap` | Clap sounds. CLAP, handclap. |
| `tom` | Tom drums. TOM, TM, PTM, DRTM, tomtom. |
| `cymbal` | Cymbals (crash, ride, splash). Not hi-hats. |
| `perc` | Handheld and world percussion: tambourine, cabasa, timbale, cowbell, congas, bongos, udu,
tam-tam, tabla, wind chimes, bell trees. Catch-all for hits that aren't a specific kit piece. |
| `drums` | Full drum loops, drum breaks, multi-piece drum patterns. Not single hits. |
| `drumhit` | Opaque AMG AKAI single drum hits without a clear kit-piece label (e.g. AMG 065SLOW9R). |
| `bass` | Bass lines, bass synths, sub bass, electric bass. BASS folder, SynthBass, TB-303/TB3, Sub Bass,
CS-80 bass, Moog Bass, Pop Bass. |
| `synth` | Synth leads, pads, sweeps, arpeggios, stabs. ARP Synths, SyncSweep, PAD, TS-SST Lead, classicrave. |
| `keys` | Electric keyboards: Rhodes, Wurlitzer, DX7, CP-70, Clavinet, OB-8 organ patch in keys context,
string machines used as pads. |
| `piano` | Acoustic or electric piano samples (not loops). Kawai/Yamaha Grand, Steinway, upright piano. |
| `organ` | Dedicated organ: Hammond B3, Vox Continental, pipe organ, Mellotron organ patches. |
| `guitar` | Guitar (acoustic or electric). Guitar Expressions, MUTE GTR, SH/GLIS/TRILL articulations. |
| `strings` | Orchestral strings: violin (VI), viola, cello (CL), bass (CB), pizzicato (PIZ), sustained (SUS),
slow attack (SLOW), string sections. |
| `brass` | Trumpet (TRP), French horn (LF = Lieblingsflöte/horn), trombone, tuba, brass sections. |
| `winds` | Woodwinds and reed instruments: flute, clarinet, saxophone, oboe, accordion, harmonica. |
| `mallet` | Mallet percussion: glockenspiel, vibraphone, xylophone, marimba, hammered dulcimer. |
| `pluck` | Plucked ethnic/world strings: koto, sitar, mandolin, banjo, cumbus, charango, ronroco, harp. |
| `vocal` | Singing, choir, vocal loops, acapellas, ah/oh pads, laughter (laughter is vocal). |
| `dialog` | Spoken voice lines from games (Half-Life, Overwatch, etc.): paths like vo/, npc/, barney/,
metropolice/, odessa/, ravenholm/. NOT singing. |
| `jawharp` | Jaw harp (Jews harp). ETHNIC/Jews Harp folder. |
| `didgeridoo` | Didgeridoo. ETHNIC/Didgeridoo folder. |
| `foley` | Environmental sounds: paper tearing, eating sounds, insects (cicadas, crickets), household
sounds, room ambience. |
| `fx` | Sound effects: lasers, explosions, impacts, game UI sounds, noise sweeps, BBC SFX library.
Also orchestral noise/aleatoric sounds wrongly classified as fx. |
| `tonal` | Generic tonal/melodic content that doesn't fit elsewhere. Use sparingly. |
| `cleave` | Cleave sounds. Use when the model says cleave and paths confirm it. |

---

## Classification rules

### 1. Paths beat the model

The model (`model_label`) is trained on audio features and is unreliable. The file path is the
best signal. When path information is unambiguous, use it regardless of model label.

Common model errors:
- `model=tom` for kicks → look for BD, KICK, BassDrum, 909BD, TX-81Z BD, iELECTRIBE-kick, MBase
- `model=clap` for hi-hats → look for HH, HAT, closedhat_, HI-HAT folder
- `model=fx` for strings → look for VI (violin), CL (cello), SUS (sustained), SLOWV, ISUSV
- `model=bass` for jaw harp → Jews Harp has a bass-frequency buzz resonance
- `model=drums` for snare → check for SN, SD, SNARE in path
- `model=guitar` for bass → check for BASS folder prefix

### 2. Agreement score

- `agreement >= 0.85` and model makes sense → high confidence for that label (or corrected label)
- `agreement 0.60–0.84` → check paths carefully, use `"med"` confidence
- `agreement < 0.50` → cluster is likely mixed; default to skip unless paths are very uniform

### 3. When to skip (use `null` label)

- Paths are opaque AKAI sample numbers with no library context (e.g. `SE_COMP/SAMPLE 7`)
- Cluster clearly contains multiple instrument types (kick + snare + hat in same cluster)
- Very low agreement (<0.40) with no strong path signal
- Mixed high-level categories (e.g. synth pad + drum loop + guitar in same cluster)

Even when skipping, include the entry in output with `"label": null` and a `"note"` explaining why.

### 4. AMG AKAI opaque hits

AMG (Advanced Media Group) hits have names like `065SLOW9R`, `108MUT2B8`, `118DIZZ2L`.
- If `model_label` is `drumhit`, `kick`, `snare`, `clap`, etc. and agreement is ≥ 0.80 → trust model
- If model says `kick` (or `snare`/`clap`) with high agreement for opaque AMG hits → use `drumhit`
  unless you can confirm the specific type from context clues (e.g. `080JUNG` = jungle kicks)
- AMG kick clusters: use `kick` if model=kick ≥ 0.85
- AMG general drum hits without confirmed type: use `drumhit`

### 5. Loop vs. one-shot

Path names often contain BPM numbers (`120BPM`, `80bpm`, `LP138`). Loops → `drums` or `bass` or
`synth` depending on content. One-shots have no BPM or say `ONE SHOT` / `ONESHOT`.

### 6. ETHNIC folder

Everything under `ETHNIC/` is an ethnic/world instrument. Check the subfolder:
- `ETHNIC/Jews Harp*` → `jawharp`
- `ETHNIC/Didgeridoo*` → `didgeridoo`
- `ETHNIC/Koto*`, `ETHNIC/Sitar*`, `ETHNIC/Mandolin*`, `ETHNIC/Banjo*` → `pluck`
- `ETHNIC/Hmrd Dulcimer*`, `ETHNIC/Glockenspiel*` → `mallet`
- `ETHNIC/Stereo Accordian*` → `winds`
- `ETHNIC/Roland SPD-20/PERCTABLA*` → `perc` (tabla)
- `ETHNIC/African Drums*`, `ETHNIC/World Whistles*` → `perc`

### 7. Half-Life / game voice lines

Paths like `vo/`, `npc/`, `barney/`, `metropolice/`, `odessa/`, `ravenholm/`, `streetwar/`,
`coast/cardock/`, `alyx_*` → `dialog` (not `vocal`)

### 8. Pizzicato strings

`PIZ`, `PIZM`, `PIZF`, `PIBF` in string library paths = pizzicato → still `strings`

---

## Confidence guidelines

| Condition | Confidence |
|-----------|------------|
| All paths clearly name the instrument + model agrees | `high` |
| Paths clearly name instrument but model disagrees | `high` (path wins) |
| Most paths match instrument, a few are opaque | `med` |
| Model agrees ≥ 0.85 with opaque AMG hits only | `med` |
| Paths mixed but one type dominates | `med` |
| Only model label to go on, agreement < 0.70 | `low` (will be skipped at default threshold) |

---

## Example batch output

```json
[
  {
    "id": 2608,
    "label": "jawharp",
    "confidence": "high",
    "note": "ETHNIC/Jews Harp&Spoons/Jews Harp 22 — all jaw harp; model=bass wrong (bass-freq buzz)"
  },
  {
    "id": 3148,
    "label": "foley",
    "confidence": "high",
    "note": "brown-fast-food-paper-bag-tear + tearinglettuce + eating-a-rusk — eating/tearing foley"
  },
  {
    "id": 4391,
    "label": "kick",
    "confidence": "high",
    "note": "Serum Assets Kick 12/17 + Big Boi Kick + XX Large Killer 909BD12 — all kicks; model=tom wrong"
  },
  {
    "id": 1706,
    "label": null,
    "confidence": "low",
    "note": "skip: akai SYNTHFACTORY SAMPLE 105/101 — completely opaque, no instrument signal"
  }
]
```

---

## Adding new labels

If you encounter an instrument not in the taxonomy (e.g. a new ethnic instrument), you should:
1. Use the closest existing label if reasonable (e.g. `perc` for new percussion, `pluck` for new
   plucked strings)
2. Or propose a new label in your note and use a placeholder like `null` for now

The DB admin can add new labels via:
```sql
INSERT OR IGNORE INTO labels(name, created_at) VALUES('newlabel', unixepoch());
```
Then re-run the apply script. `cluster_apply.py` does NOT validate labels against the labels table —
it writes whatever string you provide to `human_instrument`.

---

## Coverage tracking

```bash
sqlite3 samples.db "
  SELECT
    COUNT(*) FILTER(WHERE human_instrument IS NOT NULL) AS labeled,
    COUNT(*) AS total,
    ROUND(100.0 * COUNT(*) FILTER(WHERE human_instrument IS NOT NULL) / COUNT(*), 1) AS pct
  FROM samples WHERE cluster_id IS NOT NULL;
"
```

Regenerate dump after each batch to get fresh `n_unlab` counts:
```bash
python3 scripts/cluster_dump.py --db samples.db --unlabeled-only --output /tmp/cluster_dump.json
```
