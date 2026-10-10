# Traffic on bigger shapes (after the fact, not in the loop)

Extra shapes: K=512 M=512 N=2048, K=256 M=1024 N=1024. Bars: L5 1.6x, L6 1.25x, L7 1.05x the byte floor.

| level | kernel | shape | correct | bytes / floor | L5 | L6 | L7 | its own bar |
|---|---|---|---|---|---|---|---|---|
| 5 | ref4_L5.py | K=512 M=512 N=2048 | yes | 2.67x | over | over | over | **not met** |
| 5 | ref4_L5.py | K=256 M=1024 N=1024 | yes | 2.33x | over | over | over | **not met** |
| 5 | hoist_L5.py | K=512 M=512 N=2048 | yes | 1.33x | ok | over | over | met |
| 5 | hoist_L5.py | K=256 M=1024 N=1024 | yes | 1.17x | ok | ok | over | met |
| 6 | hoist_L6.py | K=512 M=512 N=2048 | yes | 1.33x | ok | over | over | **not met** |
| 6 | hoist_L6.py | K=256 M=1024 N=1024 | yes | 1.17x | ok | ok | over | met |
| 7 | hoist_L7.py | K=512 M=512 N=2048 | yes | 1.33x | ok | over | over | **not met** |
| 7 | hoist_L7.py | K=256 M=1024 N=1024 | yes | 1.17x | ok | ok | over | **not met** |

Controls for `scripts/traffic_check.py` (levels 5-7 fixtures in this folder, plus level 4's reference renamed as a
level-5 kernel, `/tmp/l57/ref4_L5.py` when run). Measured multiples equal the counts derived by hand (level 4's
reference reads lhsT once per N block and rhs once per M block: 2.67x and 2.33x; the hoisted kernel reads rhs once:
1.33x and 1.17x). Positive control: level 4's reference fails level 5's bar on both shapes. Negative control: the
hoisted kernel meets it. Also shown: the hoisted kernel meets level 6's bar on the loop's shapes (1.14x) and on
K=256 M=1024 N=1024 (1.17x), but not on K=512 M=512 N=2048 (1.33x > 1.25x). A level's pass depends on the shapes
it is measured on, which is why this check reports each shape.
