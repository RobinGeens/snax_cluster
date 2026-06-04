# SNAX batch-run report

_Updated 2026-06-04 10:04:46 · 4 jobs · 🧱 2 BUILD_FAIL · ✅ 2 PASS_

| App               | seqLen | dModel | n_tiles | Params                                    | Batch run       | Status        | Errors | SimbaCore |  Total |
|:------------------|-------:|-------:|--------:|:------------------------------------------|:----------------|:--------------|-------:|----------:|-------:|
| batchnorm         |    192 |      — |       — | `channels=384`                            | 20260604_100318 | ✅ PASS       |      0 |         2 | 14,929 |
| fft-tiled         |    512 |     96 |       4 | `L1=32`, `L2=16`, `nb_tiles_B=2`          | 20260604_100318 | ✅ PASS       |      0 |    25,427 | 46,035 |
| main-tiled-oscore |    192 |    384 |       8 | `dtRank=24`, `nb_l_tiles=8`, `nb_slots=2` | 20260604_100318 | 🧱 BUILD_FAIL |      — |         — |      — |
| main-tiled-oscore |    192 |    384 |       8 | `dtRank=24`, `nb_l_tiles=8`, `nb_slots=4` | 20260604_100318 | 🧱 BUILD_FAIL |      — |         — |      — |
