# Development tools

Nothing in this folder is needed to run StemDeck. It documents how the key
model was trained and how the accuracy numbers in the main README were
measured, so both can be reproduced.

## Data (all public)

| Set | Use | Source |
|---|---|---|
| Beatport EDM Key Dataset, 1,486 two-minute excerpts | training | zenodo.org/records/1101082 |
| GiantSteps+ EDM Key Dataset, 600 excerpts with corrected labels | held-out test | zenodo.org/records/1095691 |
| GiantSteps tempo set, 661 excerpts, Schreiber & Müller 2018 annotations | tempo test | zenodo.org/records/3553625 (labels), audio from cp.jku.at |

## Key model

```bash
python tools/key_model/extract.py  BEATPORT_AUDIO_DIR   bp_feats.npz
python tools/key_model/extract.py  GIANTSTEPS_AUDIO_DIR gs_feats.npz
python tools/key_model/baseline.py gs_feats.npz GIANTSTEPS_LABEL_DIR      # classic template matching
python tools/key_model/train.py --train bp_feats.npz BEATPORT_LABEL_DIR \
                                --test  gs_feats.npz GIANTSTEPS_LABEL_DIR \
                                --epochs 80 --hidden 512,256 --bank linear \
                                --also-train-on-test --out stemdeck/analysis/key_model.npz
```

`train.py` first trains on the Beatport set only and prints the score on
GiantSteps+ (never seen in training): that is the honest "held-out" number.
With `--also-train-on-test` it then retrains on both sets for the model that
ships, since more data makes a better model. The shipped model therefore has
no untouched test set left; quote the held-out number, not a score of the
shipped model on GiantSteps+.

`eval_stems.py` measures whether removing the drums before key detection
helps (it needs a model that was not trained on the test tracks). Result on
40 GiantSteps+ tracks separated with htdemucs: full mix 77.5% correct,
vocals+bass+other 77.5%, bass+other 70.0%. No benefit, so the app analyses
the full mix.

## Tempo

`eval_tempo.py AUDIO_DIR ANNOTATIONS.tsv` runs the beat tracker and grid fit
over an annotated set and stores the raw results for scoring.
