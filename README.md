# juggle-classifier

A learned juggle counter built on top of [Logan1904/JuggleNet](https://github.com/Logan1904/JuggleNet).

JuggleNet's per-frame YOLO ball detector + MediaPipe pose pipeline produces the features.
This project replaces the heuristic juggle counter with a LightGBM classifier trained on
per-event labels.

**Status:** Implementation in progress. See `docs/superpowers/specs/` for design and
`docs/superpowers/plans/` for the implementation plan.
