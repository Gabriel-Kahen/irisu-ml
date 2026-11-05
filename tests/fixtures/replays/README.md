# Replay fixtures

The `.hex` fixtures were authored from the independently documented replay bit
layout. They contain no bytes copied from the game or third-party replays and
may be redistributed with the clone.

Each `.hex` file is whitespace-separated hexadecimal. `padded.rpy.hex` covers
the v2.03 52-byte header and boundary bit values. `legacy.rpy.hex` covers the
20-byte historical header. The malformed files deliberately end in partial
records.

`seed41-score-probe.rpy` contains the 520-input, mode-0 replay used in original
score-parity calibration. It includes only the v2.03 header and packed inputs,
without game code or assets. The browser codec test pins its SHA-256 to
`1ce501febe8f3f6291e4b82736542179bd9808e412d38e0e1fb1c92d05797657`.
Keep this fixture tracked so clean checkouts can run the browser tests without
the ignored local capture directory.
