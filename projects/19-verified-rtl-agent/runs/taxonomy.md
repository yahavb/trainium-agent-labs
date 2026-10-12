### `runs/A-1.jsonl`

40 attempts, 26 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| compile: R1 is not a valid l-value | 9 | 0 |
| wrong output (seq) | 6 | 0 |
| wrong output (comb) | 6 | 0 |
| compile: R6 can not select part of scalar | 1 | 0 |
| compile: R8 has already been declared in this scope | 1 | 0 |
| compile: R4 requires an explicit cast | 1 | 0 |
| compile: other | 1 | 0 |
| compile: syntax error | 1 | 0 |

**Inside the 12 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| output X or Z | 4 |
| comb, counterexample | 2 |
| comb, no counterexample | 2 |
| seq, wrong later | 4 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS  **PASS, passed**
- `Prob031_dff` (seq): X  **FAIL, budget**
- `Prob033_ece241_2014_q1c` (comb): PASS  **PASS, passed**
- `Prob034_dff8` (seq): R1  **FAIL, budget**
- `Prob037_review2015_count1k` (seq): PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1  **FAIL, budget**
- `Prob044_vectorgates` (comb): comb?  **FAIL, budget**
- `Prob051_gates4` (comb): PASS  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1  **FAIL, budget**
- `Prob063_review2015_shiftcount` (seq): X  **FAIL, budget**
- `Prob064_vector3` (comb): X  **FAIL, budget**
- `Prob070_ece241_2013_q2` (comb): cex  **FAIL, budget**
- `Prob075_counter_2bc` (seq): PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): R6  **FAIL, budget**
- `Prob080_timer` (seq): seq@later  **FAIL, budget**
- `Prob081_7458` (comb): PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8  **FAIL, budget**
- `Prob090_circuit1` (comb): PASS  **PASS, passed**
- `Prob102_circuit3` (comb): cex  **FAIL, budget**
- `Prob105_rotate100` (seq): seq@later  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1  **FAIL, budget**
- `Prob110_fsm2` (seq): R1  **FAIL, budget**
- `Prob119_fsm3` (seq): R1  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R4  **FAIL, budget**
- `Prob125_kmap3` (comb): comb?  **FAIL, budget**
- `Prob130_circuit5` (comb): R1  **FAIL, budget**
- `Prob132_always_if2` (comb): PASS  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): R1  **FAIL, budget**
- `Prob144_conwaylife` (seq): syntax  **FAIL, budget**
- `Prob145_circuit8` (comb): X  **FAIL, budget**
- `Prob149_ece241_2013_q4` (seq): seq@later  **FAIL, budget**
- `Prob152_lemmings3` (seq): seq@later  **FAIL, budget**

### `runs/A-2.jsonl`

40 attempts, 27 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| compile: R1 is not a valid l-value | 10 | 0 |
| wrong output (comb) | 6 | 0 |
| wrong output (seq) | 5 | 0 |
| compile: R4 requires an explicit cast | 2 | 0 |
| compile: R8 has already been declared in this scope | 1 | 0 |
| compile: interface does not match testbench | 1 | 0 |
| compile: other | 1 | 0 |
| compile: syntax error | 1 | 0 |

**Inside the 11 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| output X or Z | 4 |
| comb, counterexample | 2 |
| comb, no counterexample | 2 |
| seq, wrong later | 3 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS  **PASS, passed**
- `Prob031_dff` (seq): X  **FAIL, budget**
- `Prob033_ece241_2014_q1c` (comb): PASS  **PASS, passed**
- `Prob034_dff8` (seq): R1  **FAIL, budget**
- `Prob037_review2015_count1k` (seq): PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1  **FAIL, budget**
- `Prob044_vectorgates` (comb): comb?  **FAIL, budget**
- `Prob051_gates4` (comb): PASS  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1  **FAIL, budget**
- `Prob063_review2015_shiftcount` (seq): X  **FAIL, budget**
- `Prob064_vector3` (comb): X  **FAIL, budget**
- `Prob070_ece241_2013_q2` (comb): cex  **FAIL, budget**
- `Prob075_counter_2bc` (seq): PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): R1  **FAIL, budget**
- `Prob080_timer` (seq): seq@later  **FAIL, budget**
- `Prob081_7458` (comb): PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8  **FAIL, budget**
- `Prob090_circuit1` (comb): compile: interface does not match testbench  **FAIL, budget**
- `Prob102_circuit3` (comb): cex  **FAIL, budget**
- `Prob105_rotate100` (seq): seq@later  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1  **FAIL, budget**
- `Prob110_fsm2` (seq): R1  **FAIL, budget**
- `Prob119_fsm3` (seq): R1  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R4  **FAIL, budget**
- `Prob125_kmap3` (comb): comb?  **FAIL, budget**
- `Prob130_circuit5` (comb): R1  **FAIL, budget**
- `Prob132_always_if2` (comb): PASS  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): R1  **FAIL, budget**
- `Prob144_conwaylife` (seq): syntax  **FAIL, budget**
- `Prob145_circuit8` (comb): X  **FAIL, budget**
- `Prob149_ece241_2013_q4` (seq): seq@later  **FAIL, budget**
- `Prob152_lemmings3` (seq): R4  **FAIL, budget**

### `runs/A-3.jsonl`

40 attempts, 26 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| compile: R1 is not a valid l-value | 9 | 0 |
| wrong output (comb) | 7 | 0 |
| wrong output (seq) | 6 | 0 |
| compile: R8 has already been declared in this scope | 1 | 0 |
| compile: R4 requires an explicit cast | 1 | 0 |
| compile: other | 1 | 0 |
| compile: syntax error | 1 | 0 |

**Inside the 13 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| output X or Z | 4 |
| comb, counterexample | 2 |
| comb, no counterexample | 3 |
| seq, wrong later | 4 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS  **PASS, passed**
- `Prob031_dff` (seq): X  **FAIL, budget**
- `Prob033_ece241_2014_q1c` (comb): PASS  **PASS, passed**
- `Prob034_dff8` (seq): R1  **FAIL, budget**
- `Prob037_review2015_count1k` (seq): PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1  **FAIL, budget**
- `Prob044_vectorgates` (comb): comb?  **FAIL, budget**
- `Prob051_gates4` (comb): PASS  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1  **FAIL, budget**
- `Prob063_review2015_shiftcount` (seq): X  **FAIL, budget**
- `Prob064_vector3` (comb): X  **FAIL, budget**
- `Prob070_ece241_2013_q2` (comb): cex  **FAIL, budget**
- `Prob075_counter_2bc` (seq): PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): R1  **FAIL, budget**
- `Prob080_timer` (seq): seq@later  **FAIL, budget**
- `Prob081_7458` (comb): PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8  **FAIL, budget**
- `Prob090_circuit1` (comb): PASS  **PASS, passed**
- `Prob102_circuit3` (comb): cex  **FAIL, budget**
- `Prob105_rotate100` (seq): seq@later  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1  **FAIL, budget**
- `Prob110_fsm2` (seq): R1  **FAIL, budget**
- `Prob119_fsm3` (seq): R1  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R4  **FAIL, budget**
- `Prob125_kmap3` (comb): comb?  **FAIL, budget**
- `Prob130_circuit5` (comb): R1  **FAIL, budget**
- `Prob132_always_if2` (comb): PASS  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): comb?  **FAIL, budget**
- `Prob144_conwaylife` (seq): syntax  **FAIL, budget**
- `Prob145_circuit8` (comb): X  **FAIL, budget**
- `Prob149_ece241_2013_q4` (seq): seq@later  **FAIL, budget**
- `Prob152_lemmings3` (seq): seq@later  **FAIL, budget**

### `runs/B-1.jsonl`

149 attempts, 129 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| wrong output (comb) | 34 | 7 |
| compile: R1 is not a valid l-value | 30 | 6 |
| wrong output (seq) | 17 | 3 |
| compile: other | 12 | 3 |
| compile: R8 has already been declared in this scope | 9 | 0 |
| compile: syntax error | 9 | 2 |
| wrong: reset behaviour | 8 | 2 |
| compile: R4 requires an explicit cast | 6 | 3 |
| compile: interface does not match testbench | 3 | 0 |
| compile: R6 can not select part of scalar | 1 | 0 |

**Inside the 59 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| copied | 12 |
| output X or Z | 11 |
| reset behaviour | 6 |
| comb, counterexample | 9 |
| comb, no counterexample | 12 |
| seq, wrong later | 9 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS  **PASS, passed**
- `Prob031_dff` (seq): X → PASS  **PASS, passed**
- `Prob033_ece241_2014_q1c` (comb): PASS  **PASS, passed**
- `Prob034_dff8` (seq): R1 → R1 → R1 (copy) → R1 → R1 (copy) → compile: other  **FAIL, budget**
- `Prob037_review2015_count1k` (seq): PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1 → PASS  **PASS, passed**
- `Prob044_vectorgates` (comb): comb? → PASS  **PASS, passed**
- `Prob051_gates4` (comb): PASS  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1 → R8 → R8 → R1 → R8 → R8  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1 → PASS  **PASS, passed**
- `Prob063_review2015_shiftcount` (seq): X → X → copy → X → X → seq@later  **FAIL, budget**
- `Prob064_vector3` (comb): X → X → copy → X → copy → comb?  **FAIL, budget**
- `Prob070_ece241_2013_q2` (comb): cex → copy → cex → copy → cex → copy  **FAIL, budget**
- `Prob075_counter_2bc` (seq): PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): syntax → syntax → syntax → R6 → PASS  **PASS, passed**
- `Prob080_timer` (seq): seq@later → copy → seq@later → seq@later → seq@later  **FAIL, cycling**
- `Prob081_7458` (comb): PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8 → R8 → R1 → R8 → R8 → R8  **FAIL, cycling**
- `Prob090_circuit1` (comb): R1 → R1 → R1 (copy) → compile: interface does not match testbench → compile: interface does not match testbench → compile: interface does not match testbench  **FAIL, cycling**
- `Prob102_circuit3` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob105_rotate100` (seq): seq@later → compile: other → compile: other → compile: other (copy) → seq@later → PASS  **PASS, passed**
- `Prob107_fsm1s` (seq): R1 → R1 → R1  **FAIL, cycling**
- `Prob110_fsm2` (seq): R1 → R1 → PASS  **PASS, passed**
- `Prob119_fsm3` (seq): R1 → R4 → R4 (copy) → R1 → R1 (copy) → R1  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R4 → R4 (copy) → R1 → R1 (copy) → R4 → R4 (copy)  **FAIL, budget**
- `Prob125_kmap3` (comb): comb? → comb? → comb? → comb?  **FAIL, cycling**
- `Prob130_circuit5` (comb): R1 → R1 (copy) → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob132_always_if2` (comb): PASS  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other → compile: other → compile: other (copy) → compile: other → compile: other → compile: other (copy)  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): comb? → comb? → comb? → comb? → comb? → copy  **FAIL, budget**
- `Prob144_conwaylife` (seq): syntax → syntax (copy) → syntax → syntax → syntax (copy) → syntax  **FAIL, budget**
- `Prob145_circuit8` (comb): X → comb? → copy → X → X → compile: other  **FAIL, budget**
- `Prob149_ece241_2013_q4` (seq): reset → reset → copy → reset → reset → copy  **FAIL, budget**
- `Prob152_lemmings3` (seq): seq@later → seq@later → copy → reset → reset → compile: other  **FAIL, budget**

### `runs/B-2.jsonl`

149 attempts, 131 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| compile: R1 is not a valid l-value | 39 | 10 |
| wrong output (seq) | 29 | 7 |
| wrong output (comb) | 29 | 7 |
| compile: R8 has already been declared in this scope | 9 | 0 |
| compile: other | 9 | 3 |
| compile: syntax error | 6 | 2 |
| compile: interface does not match testbench | 4 | 0 |
| compile: R4 requires an explicit cast | 4 | 2 |
| compile: R6 can not select part of scalar | 1 | 0 |
| wrong: reset behaviour | 1 | 0 |

**Inside the 59 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| copied | 14 |
| output X or Z | 15 |
| reset behaviour | 1 |
| comb, counterexample | 9 |
| comb, no counterexample | 7 |
| seq, wrong from start | 2 |
| seq, wrong later | 11 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS  **PASS, passed**
- `Prob031_dff` (seq): X → PASS  **PASS, passed**
- `Prob033_ece241_2014_q1c` (comb): PASS  **PASS, passed**
- `Prob034_dff8` (seq): X → X → copy → X → X  **FAIL, cycling**
- `Prob037_review2015_count1k` (seq): PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1 → PASS  **PASS, passed**
- `Prob044_vectorgates` (comb): comb? → PASS  **PASS, passed**
- `Prob051_gates4` (comb): PASS  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1 → R8 → R8 → R1 → R8 → R8  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1 → PASS  **PASS, passed**
- `Prob063_review2015_shiftcount` (seq): X → X → copy → X → X → seq@later  **FAIL, budget**
- `Prob064_vector3` (comb): X → X → copy → X → copy → comb?  **FAIL, budget**
- `Prob070_ece241_2013_q2` (comb): cex → copy → cex → copy → cex → copy  **FAIL, budget**
- `Prob075_counter_2bc` (seq): PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): R1 → R1 → PASS  **PASS, passed**
- `Prob080_timer` (seq): seq@later → seq@later → copy → seq@later → copy → seq@later  **FAIL, budget**
- `Prob081_7458` (comb): PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8 → R8 → R1 → R1 → R8 → R8  **FAIL, budget**
- `Prob090_circuit1` (comb): compile: interface does not match testbench → compile: interface does not match testbench → compile: interface does not match testbench  **FAIL, cycling**
- `Prob102_circuit3` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob105_rotate100` (seq): seq@later → compile: other → compile: other → compile: other (copy) → compile: interface does not match testbench → seq@later  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1 → R1 → R1 (copy) → R1 → R1 (copy) → R1  **FAIL, budget**
- `Prob110_fsm2` (seq): R1 → R1 → R1 (copy) → R1 → R1 (copy) → R1  **FAIL, budget**
- `Prob119_fsm3` (seq): R1 → R1 (copy) → R1 → R1 (copy) → R1 → R1  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R4 → R4 (copy) → R1 → R1 (copy) → R1 → R1 (copy)  **FAIL, budget**
- `Prob125_kmap3` (comb): comb? → comb? → comb?  **FAIL, cycling**
- `Prob130_circuit5` (comb): R1 → R1 (copy) → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob132_always_if2` (comb): PASS  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other → compile: other → compile: other (copy) → compile: other → compile: other → compile: other (copy)  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): R1 → R1 (copy) → comb? → copy → R6 → comb?  **FAIL, budget**
- `Prob144_conwaylife` (seq): syntax → syntax → syntax (copy) → syntax → syntax → syntax (copy)  **FAIL, budget**
- `Prob145_circuit8` (comb): X → copy → R8 → X → R1 → X  **FAIL, budget**
- `Prob149_ece241_2013_q4` (seq): seq@later → copy → reset → seq@later → seq@start → seq@start  **FAIL, budget**
- `Prob152_lemmings3` (seq): seq@later → copy → R4 → R4 (copy) → seq@later → copy  **FAIL, budget**

### `runs/B-3.jsonl`

146 attempts, 124 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| compile: R1 is not a valid l-value | 38 | 10 |
| wrong output (comb) | 33 | 7 |
| wrong output (seq) | 22 | 5 |
| compile: R8 has already been declared in this scope | 8 | 0 |
| compile: other | 7 | 2 |
| compile: R4 requires an explicit cast | 6 | 3 |
| compile: syntax error | 6 | 2 |
| compile: syntax error (truncated) | 2 | 1 |
| compile: interface does not match testbench | 1 | 0 |
| wrong: reset behaviour | 1 | 0 |

**Inside the 56 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| copied | 12 |
| output X or Z | 13 |
| reset behaviour | 1 |
| comb, counterexample | 10 |
| comb, no counterexample | 9 |
| seq, wrong from start | 1 |
| seq, wrong later | 10 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS  **PASS, passed**
- `Prob031_dff` (seq): X → PASS  **PASS, passed**
- `Prob033_ece241_2014_q1c` (comb): PASS  **PASS, passed**
- `Prob034_dff8` (seq): R1 → R1 → R1  **FAIL, cycling**
- `Prob037_review2015_count1k` (seq): PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1 → PASS  **PASS, passed**
- `Prob044_vectorgates` (comb): comb? → PASS  **PASS, passed**
- `Prob051_gates4` (comb): PASS  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1 → R8 → R8 → R1 → R8 → R8  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1 → PASS  **PASS, passed**
- `Prob063_review2015_shiftcount` (seq): X → X → seq@start → X → copy → X  **FAIL, budget**
- `Prob064_vector3` (comb): X → X → copy → X → copy → cex  **FAIL, budget**
- `Prob070_ece241_2013_q2` (comb): cex → copy → cex → cex → cex → compile: interface does not match testbench  **FAIL, budget**
- `Prob075_counter_2bc` (seq): PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): R1 → cex → cex → copy → comb? → PASS  **PASS, passed**
- `Prob080_timer` (seq): seq@later → copy → seq@later → seq@later → seq@later  **FAIL, cycling**
- `Prob081_7458` (comb): PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8 → R8 → R1 → R8 → R8 → PASS  **PASS, passed**
- `Prob090_circuit1` (comb): PASS  **PASS, passed**
- `Prob102_circuit3` (comb): cex → cex → cex  **FAIL, cycling**
- `Prob105_rotate100` (seq): seq@later → compile: other → compile: other → compile: other (copy) → seq@later → X  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1 → R1 → R1 (copy) → R1 → PASS  **PASS, passed**
- `Prob110_fsm2` (seq): R1 → R1 → R1 (copy) → R1 → R1 (copy) → R1  **FAIL, budget**
- `Prob119_fsm3` (seq): R1 → R1 (copy) → R1 → R1 (copy) → R1 → PASS  **PASS, passed**
- `Prob121_2014_q3bfsm` (seq): R4 → R4 (copy) → R1 → R1 (copy) → R1 → R1 (copy)  **FAIL, budget**
- `Prob125_kmap3` (comb): comb? → comb? → comb? → comb? → comb? → comb?  **FAIL, budget**
- `Prob130_circuit5` (comb): R1 → R1 (copy) → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob132_always_if2` (comb): PASS  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other → R4 → R4 (copy) → compile: other → compile: other → compile: other (copy)  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): R1 → R1 (copy) → R1 → R1 (copy) → comb? → copy  **FAIL, budget**
- `Prob144_conwaylife` (seq): syntax → syntax → syntax (copy) → syntax → syntax → syntax (copy)  **FAIL, budget**
- `Prob145_circuit8` (comb): X → copy → X → X → copy → X  **FAIL, budget**
- `Prob149_ece241_2013_q4` (seq): seq@later → copy → reset → seq@later → seq@later → copy  **FAIL, budget**
- `Prob152_lemmings3` (seq): seq@later → copy → R4 → R4 (copy) → syntax(trunc) → syntax(trunc) (copy)  **FAIL, budget**

### `runs/C-1.jsonl`

150 attempts, 130 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| wrong output (seq) | 38 | 7 |
| wrong output (comb) | 34 | 7 |
| compile: R1 is not a valid l-value | 28 | 6 |
| compile: R8 has already been declared in this scope | 7 | 1 |
| compile: interface does not match testbench | 5 | 1 |
| compile: R4 requires an explicit cast | 5 | 0 |
| compile: other | 4 | 1 |
| compile: syntax error | 4 | 1 |
| wrong: reset behaviour | 3 | 0 |
| compile: R6 can not select part of scalar | 2 | 0 |

**Inside the 75 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| copied | 14 |
| output X or Z | 9 |
| reset behaviour | 3 |
| comb, counterexample | 8 |
| comb, no counterexample | 15 |
| seq, wrong later | 26 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS  **PASS, passed**
- `Prob031_dff` (seq): X → PASS  **PASS, passed**
- `Prob033_ece241_2014_q1c` (comb): PASS  **PASS, passed**
- `Prob034_dff8` (seq): X → seq@later → R6 → X → seq@later → copy  **FAIL, budget**
- `Prob037_review2015_count1k` (seq): PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1 → R1 (copy) → R8 → seq@later → seq@later → seq@later  **FAIL, cycling**
- `Prob044_vectorgates` (comb): comb? → PASS  **PASS, passed**
- `Prob051_gates4` (comb): PASS  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1 → R1 (copy) → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1 → R8 → R1 → PASS  **PASS, passed**
- `Prob063_review2015_shiftcount` (seq): X → seq@later → seq@later → copy → X → seq@later  **FAIL, budget**
- `Prob064_vector3` (comb): X → copy → comb? → copy → comb? → copy  **FAIL, budget**
- `Prob070_ece241_2013_q2` (comb): cex → cex → cex → cex → cex  **FAIL, cycling**
- `Prob075_counter_2bc` (seq): PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): R6 → cex → copy → compile: interface does not match testbench → compile: interface does not match testbench (copy) → cex  **FAIL, budget**
- `Prob080_timer` (seq): seq@later → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, cycling**
- `Prob081_7458` (comb): PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8 → X → copy → PASS  **PASS, passed**
- `Prob090_circuit1` (comb): compile: interface does not match testbench → compile: interface does not match testbench → compile: interface does not match testbench  **FAIL, cycling**
- `Prob102_circuit3` (comb): cex → PASS  **PASS, passed**
- `Prob105_rotate100` (seq): seq@later → compile: other → compile: other → compile: other (copy) → seq@later → seq@later  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1 → R1 (copy) → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob110_fsm2` (seq): R1 → R1 (copy) → R8 → PASS  **PASS, passed**
- `Prob119_fsm3` (seq): R1 → R1 (copy) → R4 → R8 → R8 (copy) → R4  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R4 → R1 → R1 → seq@later → copy → R1  **FAIL, budget**
- `Prob125_kmap3` (comb): comb? → comb? → comb?  **FAIL, cycling**
- `Prob130_circuit5` (comb): R1 → R1 (copy) → R8 → comb? → copy → PASS  **PASS, passed**
- `Prob132_always_if2` (comb): PASS  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other → R4 → seq@later → copy → R4 → R1  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): comb? → comb? → R1 → comb? → comb? → comb?  **FAIL, cycling**
- `Prob144_conwaylife` (seq): syntax → syntax → syntax (copy) → syntax → R1 → seq@later  **FAIL, budget**
- `Prob145_circuit8` (comb): X → X → comb? → copy → comb? → comb?  **FAIL, cycling**
- `Prob149_ece241_2013_q4` (seq): reset → seq@later → seq@later → copy → reset → reset  **FAIL, budget**
- `Prob152_lemmings3` (seq): seq@later → copy → seq@later → copy → seq@later → seq@later  **FAIL, budget**

### `runs/C-2.jsonl`

153 attempts, 133 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| wrong output (comb) | 42 | 6 |
| compile: R1 is not a valid l-value | 33 | 8 |
| wrong output (seq) | 27 | 5 |
| compile: R8 has already been declared in this scope | 9 | 1 |
| compile: other | 8 | 1 |
| compile: R4 requires an explicit cast | 4 | 0 |
| compile: syntax error | 4 | 1 |
| wrong: reset behaviour | 3 | 0 |
| compile: R6 can not select part of scalar | 2 | 0 |
| compile: interface does not match testbench | 1 | 0 |

**Inside the 72 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| copied | 11 |
| output X or Z | 9 |
| reset behaviour | 3 |
| comb, counterexample | 17 |
| comb, no counterexample | 13 |
| seq, wrong later | 19 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS  **PASS, passed**
- `Prob031_dff` (seq): X → PASS  **PASS, passed**
- `Prob033_ece241_2014_q1c` (comb): PASS  **PASS, passed**
- `Prob034_dff8` (seq): R1 → R1 → R1 (copy) → compile: other → R1 → R8  **FAIL, budget**
- `Prob037_review2015_count1k` (seq): PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1 → R1 (copy) → R8 → seq@later → seq@later → seq@later  **FAIL, cycling**
- `Prob044_vectorgates` (comb): comb? → PASS  **PASS, passed**
- `Prob051_gates4` (comb): PASS  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1 → R1 (copy) → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1 → R8 → R1 → PASS  **PASS, passed**
- `Prob063_review2015_shiftcount` (seq): X → seq@later → seq@later → copy → X → seq@later  **FAIL, budget**
- `Prob064_vector3` (comb): X → X → copy → comb? → copy → comb?  **FAIL, budget**
- `Prob070_ece241_2013_q2` (comb): cex → cex → cex → cex → cex → cex  **FAIL, cycling**
- `Prob075_counter_2bc` (seq): PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): R6 → cex → cex → cex  **FAIL, cycling**
- `Prob080_timer` (seq): seq@later → seq@later → copy → seq@later → seq@later → seq@later  **FAIL, cycling**
- `Prob081_7458` (comb): PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8 → X → comb? → PASS  **PASS, passed**
- `Prob090_circuit1` (comb): compile: interface does not match testbench → R1 → cex → cex → copy → PASS  **PASS, passed**
- `Prob102_circuit3` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob105_rotate100` (seq): seq@later → compile: other → compile: other → compile: other (copy) → seq@later → copy  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1 → R1 (copy) → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob110_fsm2` (seq): R1 → R1 (copy) → R8 → PASS  **PASS, passed**
- `Prob119_fsm3` (seq): R1 → R1 (copy) → R4 → R1 → R1 (copy) → R8  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R4 → R1 → R8 → R8 (copy) → R4 → R4  **FAIL, budget**
- `Prob125_kmap3` (comb): comb? → comb? → comb?  **FAIL, cycling**
- `Prob130_circuit5` (comb): R1 → R1 (copy) → R8 → comb? → copy → PASS  **PASS, passed**
- `Prob132_always_if2` (comb): PASS  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other → R1 → compile: other → seq@later → copy → R1  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): comb? → comb? → R1 → comb? → copy → R6  **FAIL, budget**
- `Prob144_conwaylife` (seq): compile: other → syntax → compile: other → syntax → syntax → syntax (copy)  **FAIL, budget**
- `Prob145_circuit8` (comb): X → X → copy → comb? → comb? → X  **FAIL, budget**
- `Prob149_ece241_2013_q4` (seq): reset → seq@later → seq@later → copy → reset → reset  **FAIL, budget**
- `Prob152_lemmings3` (seq): seq@later → seq@later → seq@later  **FAIL, cycling**

### `runs/C-3.jsonl`

144 attempts, 123 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| wrong output (comb) | 35 | 4 |
| compile: R1 is not a valid l-value | 28 | 7 |
| wrong output (seq) | 25 | 0 |
| compile: other | 9 | 3 |
| compile: R8 has already been declared in this scope | 8 | 1 |
| compile: syntax error | 5 | 1 |
| compile: interface does not match testbench | 4 | 1 |
| compile: R4 requires an explicit cast | 4 | 0 |
| wrong: reset behaviour | 3 | 1 |
| compile: R6 can not select part of scalar | 2 | 0 |

**Inside the 63 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| copied | 5 |
| output X or Z | 10 |
| reset behaviour | 2 |
| comb, counterexample | 10 |
| comb, no counterexample | 15 |
| seq, wrong later | 21 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS  **PASS, passed**
- `Prob031_dff` (seq): X → PASS  **PASS, passed**
- `Prob033_ece241_2014_q1c` (comb): PASS  **PASS, passed**
- `Prob034_dff8` (seq): X → seq@later → R6 → X → seq@later → PASS  **PASS, passed**
- `Prob037_review2015_count1k` (seq): PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1 → R1 (copy) → R8 → seq@later → PASS  **PASS, passed**
- `Prob044_vectorgates` (comb): comb? → PASS  **PASS, passed**
- `Prob051_gates4` (comb): PASS  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1 → R1 (copy) → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1 → R8 → R1 → PASS  **PASS, passed**
- `Prob063_review2015_shiftcount` (seq): X → seq@later → seq@later → seq@later  **FAIL, cycling**
- `Prob064_vector3` (comb): X → X → X  **FAIL, cycling**
- `Prob070_ece241_2013_q2` (comb): cex → cex → cex  **FAIL, cycling**
- `Prob075_counter_2bc` (seq): PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): R1 → R8 → comb? → comb? → comb?  **FAIL, cycling**
- `Prob080_timer` (seq): seq@later → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, cycling**
- `Prob081_7458` (comb): PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8 → X → comb? → PASS  **PASS, passed**
- `Prob090_circuit1` (comb): X → compile: interface does not match testbench → compile: interface does not match testbench → compile: interface does not match testbench (copy) → cex → cex  **FAIL, budget**
- `Prob102_circuit3` (comb): cex → cex → cex → copy → cex → cex  **FAIL, budget**
- `Prob105_rotate100` (seq): seq@later → compile: other → compile: other → compile: other (copy) → seq@later → seq@later  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1 → R1 (copy) → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob110_fsm2` (seq): R1 → R1 (copy) → R1 → PASS  **PASS, passed**
- `Prob119_fsm3` (seq): R1 → R1 (copy) → R4 → R1 → R1 (copy) → R8  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R4 → R1 → R8 → R8 (copy) → R4 → R4  **FAIL, budget**
- `Prob125_kmap3` (comb): comb? → comb? → comb?  **FAIL, cycling**
- `Prob130_circuit5` (comb): R1 → R1 (copy) → R8 → comb? → copy → PASS  **PASS, passed**
- `Prob132_always_if2` (comb): PASS  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other → compile: other → compile: other (copy) → compile: other → compile: other (copy) → compile: other  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): comb? → comb? → R1 → comb? → copy → R6  **FAIL, budget**
- `Prob144_conwaylife` (seq): compile: interface does not match testbench → syntax → syntax (copy) → syntax → syntax → syntax  **FAIL, budget**
- `Prob145_circuit8` (comb): X → comb? → copy → comb? → comb?  **FAIL, cycling**
- `Prob149_ece241_2013_q4` (seq): reset → reset → copy → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob152_lemmings3` (seq): seq@later → seq@later → seq@later  **FAIL, cycling**

### `runs/D-1.jsonl`

183 attempts, 141 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| wrong output (seq) | 36 | 0 |
| compile: R1 is not a valid l-value | 36 | 3 |
| wrong output (comb) | 35 | 0 |
| compile: R8 has already been declared in this scope | 13 | 0 |
| compile: syntax error | 9 | 0 |
| wrong: reset behaviour | 4 | 0 |
| compile: other | 4 | 0 |
| compile: R4 requires an explicit cast | 2 | 0 |
| compile: R6 can not select part of scalar | 1 | 0 |
| compile: interface does not match testbench | 1 | 0 |

**Inside the 75 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| output X or Z | 19 |
| reset behaviour | 2 |
| comb, counterexample | 15 |
| comb, no counterexample | 16 |
| seq, wrong later | 23 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob031_dff` (seq): X → X → X → PASS → X → X  **PASS, passed**
- `Prob033_ece241_2014_q1c` (comb): PASS → PASS → cex  **PASS, passed**
- `Prob034_dff8` (seq): X → R8 → X → X → R8 → X  **FAIL, budget**
- `Prob037_review2015_count1k` (seq): PASS → seq@later → PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS → PASS → PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1 → R8 → R8 → R1 → R8 → PASS  **PASS, passed**
- `Prob044_vectorgates` (comb): comb? → PASS → comb?  **PASS, passed**
- `Prob051_gates4` (comb): PASS → PASS → R6  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1 → R1 → R1 → R1 (copy) → R1 (copy) → R1 (copy)  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1 → R8 → R1 → R8 → PASS → R8  **PASS, passed**
- `Prob063_review2015_shiftcount` (seq): X → X → X → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob064_vector3` (comb): X → PASS → comb?  **PASS, passed**
- `Prob070_ece241_2013_q2` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob075_counter_2bc` (seq): X → X → PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): R1 → comb? → PASS  **PASS, passed**
- `Prob080_timer` (seq): seq@later → X → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob081_7458` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8 → PASS → R1  **PASS, passed**
- `Prob090_circuit1` (comb): comb? → PASS → cex  **PASS, passed**
- `Prob102_circuit3` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob105_rotate100` (seq): seq@later → seq@later → R8 → compile: other → seq@later → seq@later  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1 → R1 → R1 → R1 → R8 → R1  **FAIL, budget**
- `Prob110_fsm2` (seq): R1 → R1 → PASS  **PASS, passed**
- `Prob119_fsm3` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R4 → R1 → R1 → R1 → R8 → seq@later  **FAIL, budget**
- `Prob125_kmap3` (comb): comb? → comb? → comb? → comb? → comb? → comb?  **FAIL, budget**
- `Prob130_circuit5` (comb): R1 → R1 → syntax → R1 → R1 → R1  **FAIL, budget**
- `Prob132_always_if2` (comb): PASS → PASS → cex  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other → R1 → compile: other → R1 → seq@later → compile: other  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): comb? → R1 → syntax → comb? → comb? → syntax  **FAIL, budget**
- `Prob144_conwaylife` (seq): syntax → syntax → syntax → syntax → syntax → syntax  **FAIL, budget**
- `Prob145_circuit8` (comb): X → R8 → comb? → X → X → comb?  **FAIL, budget**
- `Prob149_ece241_2013_q4` (seq): seq@later → compile: interface does not match testbench → reset → seq@later → seq@later → reset  **FAIL, budget**
- `Prob152_lemmings3` (seq): seq@later → seq@later → seq@later → R4 → seq@later → seq@later  **FAIL, budget**

### `runs/D-2.jsonl`

186 attempts, 142 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| wrong output (seq) | 37 | 0 |
| compile: R1 is not a valid l-value | 37 | 3 |
| wrong output (comb) | 35 | 0 |
| compile: R8 has already been declared in this scope | 11 | 0 |
| compile: syntax error | 8 | 0 |
| wrong: reset behaviour | 5 | 0 |
| compile: R6 can not select part of scalar | 3 | 0 |
| compile: other | 3 | 0 |
| compile: R4 requires an explicit cast | 3 | 0 |

**Inside the 77 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| output X or Z | 18 |
| reset behaviour | 3 |
| comb, counterexample | 16 |
| comb, no counterexample | 14 |
| seq, wrong from start | 1 |
| seq, wrong later | 25 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob031_dff` (seq): X → X → X → PASS → X → X  **PASS, passed**
- `Prob033_ece241_2014_q1c` (comb): PASS → PASS → cex  **PASS, passed**
- `Prob034_dff8` (seq): X → R8 → X → X → R8 → seq@later  **FAIL, budget**
- `Prob037_review2015_count1k` (seq): PASS → PASS → PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS → PASS → PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1 → seq@later → R8 → R1 → seq@later → PASS  **PASS, passed**
- `Prob044_vectorgates` (comb): comb? → PASS → comb?  **PASS, passed**
- `Prob051_gates4` (comb): PASS → PASS → R6  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1 → R1 → R1 → R1 (copy) → R1 (copy) → R1 (copy)  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1 → R8 → R1 → R8 → PASS → R8  **PASS, passed**
- `Prob063_review2015_shiftcount` (seq): X → X → X → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob064_vector3` (comb): X → cex → comb? → X → cex → comb?  **FAIL, budget**
- `Prob070_ece241_2013_q2` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob075_counter_2bc` (seq): X → X → PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): R6 → R6 → PASS  **PASS, passed**
- `Prob080_timer` (seq): seq@later → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob081_7458` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8 → PASS → R1  **PASS, passed**
- `Prob090_circuit1` (comb): PASS → PASS → cex  **PASS, passed**
- `Prob102_circuit3` (comb): cex → cex → cex → PASS → cex → cex  **PASS, passed**
- `Prob105_rotate100` (seq): seq@later → seq@later → seq@later → compile: other → seq@later → compile: other  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1 → R1 → R1 → R1 → R8 → R1  **FAIL, budget**
- `Prob110_fsm2` (seq): R1 → R1 → PASS  **PASS, passed**
- `Prob119_fsm3` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R4 → R1 → R1 → R1 → R8 → seq@later  **FAIL, budget**
- `Prob125_kmap3` (comb): comb? → comb? → comb? → comb? → comb? → comb?  **FAIL, budget**
- `Prob130_circuit5` (comb): R1 → R1 → R1 → R1 → R1 → R8  **FAIL, budget**
- `Prob132_always_if2` (comb): PASS → PASS → cex  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other → R1 → R1 → R1 → seq@later → seq@later  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): R1 → comb? → syntax → R1 → comb? → syntax  **FAIL, budget**
- `Prob144_conwaylife` (seq): syntax → syntax → syntax → syntax → syntax → syntax  **FAIL, budget**
- `Prob145_circuit8` (comb): X → R8 → comb? → X → X → comb?  **FAIL, budget**
- `Prob149_ece241_2013_q4` (seq): seq@later → seq@later → reset → seq@later → seq@later → reset  **FAIL, budget**
- `Prob152_lemmings3` (seq): R4 → reset → seq@later → R4 → seq@start → seq@later  **FAIL, budget**

### `runs/D-3.jsonl`

183 attempts, 139 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| compile: R1 is not a valid l-value | 37 | 3 |
| wrong output (comb) | 36 | 0 |
| wrong output (seq) | 33 | 0 |
| compile: R8 has already been declared in this scope | 9 | 0 |
| compile: syntax error | 8 | 0 |
| compile: other | 7 | 0 |
| wrong: reset behaviour | 7 | 0 |
| compile: R6 can not select part of scalar | 1 | 0 |
| compile: R4 requires an explicit cast | 1 | 0 |

**Inside the 76 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| output X or Z | 19 |
| reset behaviour | 6 |
| comb, counterexample | 14 |
| comb, no counterexample | 15 |
| seq, wrong later | 22 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob031_dff` (seq): X → X → X → PASS → X → X  **PASS, passed**
- `Prob033_ece241_2014_q1c` (comb): PASS → PASS → cex  **PASS, passed**
- `Prob034_dff8` (seq): X → R8 → X → seq@later → R8 → X  **FAIL, budget**
- `Prob037_review2015_count1k` (seq): PASS → seq@later → PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS → PASS → PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1 → PASS → R1  **PASS, passed**
- `Prob044_vectorgates` (comb): comb? → PASS → comb?  **PASS, passed**
- `Prob051_gates4` (comb): PASS → PASS → R6  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1 → R1 → R1 → R1 (copy) → R1 (copy) → R1 (copy)  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1 → R8 → R1 → R8 → PASS → R8  **PASS, passed**
- `Prob063_review2015_shiftcount` (seq): X → X → X → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob064_vector3` (comb): X → X → comb? → X → compile: other → X  **FAIL, budget**
- `Prob070_ece241_2013_q2` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob075_counter_2bc` (seq): PASS → X → PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): R1 → R1 → PASS  **PASS, passed**
- `Prob080_timer` (seq): seq@later → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob081_7458` (comb): PASS → PASS → PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8 → PASS → R1  **PASS, passed**
- `Prob090_circuit1` (comb): PASS → PASS → cex  **PASS, passed**
- `Prob102_circuit3` (comb): cex → cex → cex → PASS → cex → cex  **PASS, passed**
- `Prob105_rotate100` (seq): seq@later → seq@later → seq@later → compile: other → compile: other → compile: other  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1 → R1 → R1 → R1 → R8 → R1  **FAIL, budget**
- `Prob110_fsm2` (seq): R1 → R1 → PASS  **PASS, passed**
- `Prob119_fsm3` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R4 → R1 → R1 → R1 → R8 → seq@later  **FAIL, budget**
- `Prob125_kmap3` (comb): comb? → comb? → comb? → comb? → comb? → comb?  **FAIL, budget**
- `Prob130_circuit5` (comb): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob132_always_if2` (comb): PASS → PASS → cex  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other → R1 → compile: other → R1 → seq@later → compile: other  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): comb? → comb? → syntax → comb? → comb? → syntax  **FAIL, budget**
- `Prob144_conwaylife` (seq): syntax → syntax → syntax → syntax → syntax → syntax  **FAIL, budget**
- `Prob145_circuit8` (comb): X → R8 → comb? → X → X → comb?  **FAIL, budget**
- `Prob149_ece241_2013_q4` (seq): reset → reset → reset → reset → reset → reset  **FAIL, budget**
- `Prob152_lemmings3` (seq): seq@later → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**

### `runs/Q-1.jsonl`

159 attempts, 140 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| compile: R1 is not a valid l-value | 42 | 0 |
| wrong output (comb) | 34 | 0 |
| wrong output (seq) | 31 | 0 |
| compile: R8 has already been declared in this scope | 9 | 0 |
| compile: syntax error | 7 | 0 |
| compile: other | 5 | 0 |
| wrong: reset behaviour | 5 | 0 |
| compile: R6 can not select part of scalar | 3 | 0 |
| no module TopModule | 2 | 0 |
| compile: interface does not match testbench | 1 | 0 |
| compile: R4 requires an explicit cast | 1 | 0 |

**Inside the 70 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| output X or Z | 21 |
| reset behaviour | 5 |
| comb, counterexample | 14 |
| comb, no counterexample | 14 |
| seq, wrong later | 16 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS  **PASS, passed**
- `Prob031_dff` (seq): X → X → X → X → X → X  **FAIL, budget**
- `Prob033_ece241_2014_q1c` (comb): PASS  **PASS, passed**
- `Prob034_dff8` (seq): R1 → R8 → X → X → compile: other → X  **FAIL, budget**
- `Prob037_review2015_count1k` (seq): PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1 → R8 → R8 → R1 → R8 → R1  **FAIL, budget**
- `Prob044_vectorgates` (comb): comb? → PASS  **PASS, passed**
- `Prob051_gates4` (comb): PASS  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1 → R8 → R1 → R1 → PASS  **PASS, passed**
- `Prob063_review2015_shiftcount` (seq): X → X → X → X → X → X  **FAIL, budget**
- `Prob064_vector3` (comb): X → X → comb? → X → cex → comb?  **FAIL, budget**
- `Prob070_ece241_2013_q2` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob075_counter_2bc` (seq): PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): R1 → R6 → cex → R1 → R6 → PASS  **PASS, passed**
- `Prob080_timer` (seq): seq@later → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob081_7458` (comb): PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8 → PASS  **PASS, passed**
- `Prob090_circuit1` (comb): compile: interface does not match testbench → PASS  **PASS, passed**
- `Prob102_circuit3` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob105_rotate100` (seq): seq@later → seq@later → R8 → seq@later → seq@later → R8  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob110_fsm2` (seq): R1 → R1 → PASS  **PASS, passed**
- `Prob119_fsm3` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R4 → R8 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob125_kmap3` (comb): comb? → comb? → comb? → comb? → comb? → comb?  **FAIL, budget**
- `Prob130_circuit5` (comb): R1 → R1 → comb? → R1 → R1 → R1  **FAIL, budget**
- `Prob132_always_if2` (comb): PASS  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other → R1 → compile: other → compile: other → R1 → compile: other  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): R1 → R1 → syntax → comb? → comb? → R6  **FAIL, budget**
- `Prob144_conwaylife` (seq): syntax → syntax → syntax → syntax → syntax → syntax  **FAIL, budget**
- `Prob145_circuit8` (comb): X → no module TopModule → comb? → X → X → comb?  **FAIL, budget**
- `Prob149_ece241_2013_q4` (seq): seq@later → no module TopModule → reset → reset → reset → reset  **FAIL, budget**
- `Prob152_lemmings3` (seq): seq@later → seq@later → seq@later → seq@later → reset → seq@later  **FAIL, budget**

### `runs/Q-2.jsonl`

152 attempts, 132 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| compile: R1 is not a valid l-value | 42 | 0 |
| wrong output (seq) | 33 | 0 |
| wrong output (comb) | 27 | 0 |
| compile: R8 has already been declared in this scope | 8 | 0 |
| compile: syntax error | 7 | 0 |
| compile: other | 5 | 0 |
| wrong: reset behaviour | 4 | 0 |
| compile: R6 can not select part of scalar | 2 | 0 |
| compile: R4 requires an explicit cast | 2 | 0 |
| compile: interface does not match testbench | 1 | 0 |
| no module TopModule | 1 | 0 |

**Inside the 64 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| output X or Z | 18 |
| reset behaviour | 3 |
| comb, counterexample | 12 |
| comb, no counterexample | 12 |
| seq, wrong later | 19 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS  **PASS, passed**
- `Prob031_dff` (seq): X → X → X → X → X → X  **FAIL, budget**
- `Prob033_ece241_2014_q1c` (comb): PASS  **PASS, passed**
- `Prob034_dff8` (seq): R1 → compile: other → X → R1 → R8 → X  **FAIL, budget**
- `Prob037_review2015_count1k` (seq): PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1 → R8 → R1 → R1 → PASS  **PASS, passed**
- `Prob044_vectorgates` (comb): comb? → PASS  **PASS, passed**
- `Prob051_gates4` (comb): PASS  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1 → R8 → R1 → R1 → R8 → R1  **FAIL, budget**
- `Prob063_review2015_shiftcount` (seq): X → X → X → X → X → X  **FAIL, budget**
- `Prob064_vector3` (comb): X → PASS  **PASS, passed**
- `Prob070_ece241_2013_q2` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob075_counter_2bc` (seq): PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): comb? → R6 → PASS  **PASS, passed**
- `Prob080_timer` (seq): seq@later → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob081_7458` (comb): PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8 → PASS  **PASS, passed**
- `Prob090_circuit1` (comb): compile: interface does not match testbench → PASS  **PASS, passed**
- `Prob102_circuit3` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob105_rotate100` (seq): seq@later → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob110_fsm2` (seq): R1 → R1 → PASS  **PASS, passed**
- `Prob119_fsm3` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R4 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob125_kmap3` (comb): comb? → comb? → comb? → comb? → comb? → comb?  **FAIL, budget**
- `Prob130_circuit5` (comb): R1 → no module TopModule → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob132_always_if2` (comb): PASS  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other → R1 → compile: other → compile: other → R1 → compile: other  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): comb? → comb? → R6 → R1 → R8 → syntax  **FAIL, budget**
- `Prob144_conwaylife` (seq): syntax → syntax → syntax → syntax → syntax → syntax  **FAIL, budget**
- `Prob145_circuit8` (comb): X → R8 → comb? → X → R8 → comb?  **FAIL, budget**
- `Prob149_ece241_2013_q4` (seq): seq@later → seq@later → reset → reset → X → reset  **FAIL, budget**
- `Prob152_lemmings3` (seq): R4 → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**

### `runs/Q-3.jsonl`

157 attempts, 139 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| compile: R1 is not a valid l-value | 45 | 0 |
| wrong output (seq) | 35 | 0 |
| wrong output (comb) | 30 | 0 |
| compile: R8 has already been declared in this scope | 10 | 0 |
| compile: syntax error | 8 | 0 |
| compile: other | 4 | 0 |
| wrong: reset behaviour | 3 | 0 |
| compile: interface does not match testbench | 2 | 0 |
| compile: R4 requires an explicit cast | 2 | 0 |

**Inside the 68 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| output X or Z | 21 |
| reset behaviour | 3 |
| comb, counterexample | 13 |
| comb, no counterexample | 12 |
| seq, wrong later | 19 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS  **PASS, passed**
- `Prob031_dff` (seq): X → X → X → X → X → X  **FAIL, budget**
- `Prob033_ece241_2014_q1c` (comb): PASS  **PASS, passed**
- `Prob034_dff8` (seq): R1 → X → X → X → R8 → X  **FAIL, budget**
- `Prob037_review2015_count1k` (seq): PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1 → R8 → R8 → R1 → R8 → R8  **FAIL, budget**
- `Prob044_vectorgates` (comb): comb? → PASS  **PASS, passed**
- `Prob051_gates4` (comb): PASS  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1 → R8 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob063_review2015_shiftcount` (seq): X → X → X → X → X → X  **FAIL, budget**
- `Prob064_vector3` (comb): X → cex → comb? → X → X → comb?  **FAIL, budget**
- `Prob070_ece241_2013_q2` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob075_counter_2bc` (seq): PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): R1 → R1 → PASS  **PASS, passed**
- `Prob080_timer` (seq): seq@later → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob081_7458` (comb): PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8 → PASS  **PASS, passed**
- `Prob090_circuit1` (comb): compile: interface does not match testbench → PASS  **PASS, passed**
- `Prob102_circuit3` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob105_rotate100` (seq): seq@later → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob110_fsm2` (seq): R1 → R1 → PASS  **PASS, passed**
- `Prob119_fsm3` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R4 → R8 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob125_kmap3` (comb): comb? → comb? → comb? → comb? → comb? → comb?  **FAIL, budget**
- `Prob130_circuit5` (comb): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob132_always_if2` (comb): PASS  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other → R1 → compile: other → compile: other → R1 → compile: other  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): R1 → comb? → syntax → R1 → R1 → syntax  **FAIL, budget**
- `Prob144_conwaylife` (seq): syntax → syntax → syntax → syntax → syntax → syntax  **FAIL, budget**
- `Prob145_circuit8` (comb): X → R8 → comb? → X → R8 → comb?  **FAIL, budget**
- `Prob149_ece241_2013_q4` (seq): seq@later → compile: interface does not match testbench → reset → reset → seq@later → reset  **FAIL, budget**
- `Prob152_lemmings3` (seq): seq@later → seq@later → seq@later → R4 → seq@later → seq@later  **FAIL, budget**

### `runs/Q-4.jsonl`

160 attempts, 142 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| compile: R1 is not a valid l-value | 42 | 0 |
| wrong output (seq) | 33 | 0 |
| wrong output (comb) | 32 | 0 |
| compile: R8 has already been declared in this scope | 10 | 0 |
| compile: syntax error | 6 | 0 |
| compile: R6 can not select part of scalar | 5 | 0 |
| wrong: reset behaviour | 5 | 0 |
| compile: other | 4 | 0 |
| compile: interface does not match testbench | 2 | 0 |
| compile: R4 requires an explicit cast | 2 | 0 |
| no module TopModule | 1 | 0 |

**Inside the 70 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| output X or Z | 23 |
| reset behaviour | 5 |
| comb, counterexample | 13 |
| comb, no counterexample | 13 |
| seq, wrong later | 16 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS  **PASS, passed**
- `Prob031_dff` (seq): X → X → X → X → X → X  **FAIL, budget**
- `Prob033_ece241_2014_q1c` (comb): PASS  **PASS, passed**
- `Prob034_dff8` (seq): X → R8 → X → R1 → R8 → X  **FAIL, budget**
- `Prob037_review2015_count1k` (seq): PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1 → R1 → R8 → R1 → seq@later → R1  **FAIL, budget**
- `Prob044_vectorgates` (comb): comb? → PASS  **PASS, passed**
- `Prob051_gates4` (comb): PASS  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1 → R8 → R1 → R1 → R8 → R1  **FAIL, budget**
- `Prob063_review2015_shiftcount` (seq): X → X → X → X → X → X  **FAIL, budget**
- `Prob064_vector3` (comb): X → X → comb? → X → X → comb?  **FAIL, budget**
- `Prob070_ece241_2013_q2` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob075_counter_2bc` (seq): PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): R1 → R6 → cex → R6 → R6 → PASS  **PASS, passed**
- `Prob080_timer` (seq): seq@later → seq@later → seq@later → seq@later → X → seq@later  **FAIL, budget**
- `Prob081_7458` (comb): PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8 → PASS  **PASS, passed**
- `Prob090_circuit1` (comb): compile: interface does not match testbench → PASS  **PASS, passed**
- `Prob102_circuit3` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob105_rotate100` (seq): seq@later → R8 → R8 → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob110_fsm2` (seq): R1 → R1 → PASS  **PASS, passed**
- `Prob119_fsm3` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R4 → R1 → R1 → R4 → R1 → R1  **FAIL, budget**
- `Prob125_kmap3` (comb): comb? → comb? → comb? → comb? → comb? → comb?  **FAIL, budget**
- `Prob130_circuit5` (comb): R1 → R1 → R6 → R1 → no module TopModule → R1  **FAIL, budget**
- `Prob132_always_if2` (comb): PASS  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other → R1 → compile: other → compile: other → R1 → compile: other  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): comb? → R1 → syntax → R1 → comb? → R6  **FAIL, budget**
- `Prob144_conwaylife` (seq): X → syntax → syntax → syntax → syntax → syntax  **FAIL, budget**
- `Prob145_circuit8` (comb): X → R8 → comb? → X → R8 → comb?  **FAIL, budget**
- `Prob149_ece241_2013_q4` (seq): reset → compile: interface does not match testbench → reset → seq@later → reset → reset  **FAIL, budget**
- `Prob152_lemmings3` (seq): seq@later → reset → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**

### `runs/R-1.jsonl`

171 attempts, 157 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| compile: R1 is not a valid l-value | 55 | 0 |
| wrong output (comb) | 39 | 0 |
| wrong output (seq) | 33 | 0 |
| compile: syntax error | 7 | 0 |
| compile: R8 has already been declared in this scope | 6 | 0 |
| compile: R4 requires an explicit cast | 6 | 0 |
| compile: other | 6 | 0 |
| wrong: reset behaviour | 4 | 0 |
| compile: R6 can not select part of scalar | 1 | 0 |

**Inside the 76 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| output X or Z | 26 |
| reset behaviour | 3 |
| comb, counterexample | 12 |
| comb, no counterexample | 15 |
| seq, wrong later | 20 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS  **PASS, passed**
- `Prob031_dff` (seq): X → X → X → X → X → X  **FAIL, budget**
- `Prob033_ece241_2014_q1c` (comb): PASS  **PASS, passed**
- `Prob034_dff8` (seq): R1 → X → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob037_review2015_count1k` (seq): PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob044_vectorgates` (comb): comb? → comb? → comb? → comb? → comb? → comb?  **FAIL, budget**
- `Prob051_gates4` (comb): PASS  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob063_review2015_shiftcount` (seq): X → X → X → X → X → X  **FAIL, budget**
- `Prob064_vector3` (comb): X → X → X → X → X → X  **FAIL, budget**
- `Prob070_ece241_2013_q2` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob075_counter_2bc` (seq): X → PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): R1 → R1 → R6 → R1 → R1 → R1  **FAIL, budget**
- `Prob080_timer` (seq): seq@later → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob081_7458` (comb): PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8 → R8 → R8 → R8 → R8 → R8  **FAIL, budget**
- `Prob090_circuit1` (comb): PASS  **PASS, passed**
- `Prob102_circuit3` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob105_rotate100` (seq): seq@later → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob110_fsm2` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob119_fsm3` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R4 → R4 → R4 → R4 → R4 → R4  **FAIL, budget**
- `Prob125_kmap3` (comb): comb? → comb? → comb? → comb? → comb? → comb?  **FAIL, budget**
- `Prob130_circuit5` (comb): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob132_always_if2` (comb): PASS  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other → compile: other → compile: other → compile: other → compile: other → compile: other  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): R1 → comb? → R1 → R1 → comb? → comb?  **FAIL, budget**
- `Prob144_conwaylife` (seq): syntax → syntax → syntax → syntax → syntax → syntax  **FAIL, budget**
- `Prob145_circuit8` (comb): X → X → X → X → X → X  **FAIL, budget**
- `Prob149_ece241_2013_q4` (seq): seq@later → seq@later → reset → syntax → reset → reset  **FAIL, budget**
- `Prob152_lemmings3` (seq): seq@later → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**

### `runs/R-2.jsonl`

175 attempts, 162 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| compile: R1 is not a valid l-value | 57 | 0 |
| wrong output (comb) | 41 | 0 |
| wrong output (seq) | 34 | 0 |
| compile: R4 requires an explicit cast | 7 | 0 |
| compile: R8 has already been declared in this scope | 6 | 0 |
| compile: other | 6 | 0 |
| compile: syntax error | 6 | 0 |
| wrong: reset behaviour | 3 | 0 |
| compile: interface does not match testbench | 2 | 0 |

**Inside the 78 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| output X or Z | 28 |
| reset behaviour | 3 |
| comb, counterexample | 12 |
| comb, no counterexample | 17 |
| seq, wrong later | 18 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS  **PASS, passed**
- `Prob031_dff` (seq): X → X → X → X → X → X  **FAIL, budget**
- `Prob033_ece241_2014_q1c` (comb): PASS  **PASS, passed**
- `Prob034_dff8` (seq): X → X → X → R1 → R1 → X  **FAIL, budget**
- `Prob037_review2015_count1k` (seq): PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob044_vectorgates` (comb): comb? → comb? → comb? → comb? → comb? → comb?  **FAIL, budget**
- `Prob051_gates4` (comb): PASS  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob063_review2015_shiftcount` (seq): X → X → X → X → X → X  **FAIL, budget**
- `Prob064_vector3` (comb): X → X → X → X → X → X  **FAIL, budget**
- `Prob070_ece241_2013_q2` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob075_counter_2bc` (seq): PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob080_timer` (seq): seq@later → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob081_7458` (comb): PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8 → R8 → R8 → R8 → R8 → R8  **FAIL, budget**
- `Prob090_circuit1` (comb): compile: interface does not match testbench → comb? → compile: interface does not match testbench → comb? → R1 → R1  **FAIL, budget**
- `Prob102_circuit3` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob105_rotate100` (seq): seq@later → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob110_fsm2` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob119_fsm3` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R1 → R4 → R4 → R4 → R1 → R4  **FAIL, budget**
- `Prob125_kmap3` (comb): comb? → comb? → comb? → comb? → comb? → comb?  **FAIL, budget**
- `Prob130_circuit5` (comb): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob132_always_if2` (comb): PASS  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other → compile: other → compile: other → compile: other → compile: other → compile: other  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): R1 → R1 → R1 → comb? → comb? → comb?  **FAIL, budget**
- `Prob144_conwaylife` (seq): syntax → syntax → syntax → syntax → syntax → syntax  **FAIL, budget**
- `Prob145_circuit8` (comb): X → X → X → X → X → X  **FAIL, budget**
- `Prob149_ece241_2013_q4` (seq): seq@later → seq@later → reset → reset → reset → seq@later  **FAIL, budget**
- `Prob152_lemmings3` (seq): seq@later → R4 → R4 → seq@later → seq@later → R4  **FAIL, budget**

### `runs/R-3.jsonl`

170 attempts, 156 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| compile: R1 is not a valid l-value | 48 | 0 |
| wrong output (comb) | 41 | 0 |
| wrong output (seq) | 35 | 0 |
| compile: R4 requires an explicit cast | 7 | 0 |
| compile: other | 7 | 0 |
| compile: R8 has already been declared in this scope | 6 | 0 |
| compile: syntax error | 5 | 0 |
| wrong: reset behaviour | 5 | 0 |
| compile: R6 can not select part of scalar | 2 | 0 |

**Inside the 81 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| output X or Z | 29 |
| reset behaviour | 5 |
| comb, counterexample | 12 |
| comb, no counterexample | 17 |
| seq, wrong later | 18 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob007_wire` (comb): PASS  **PASS, passed**
- `Prob010_mt2015_q4a` (comb): PASS  **PASS, passed**
- `Prob015_vector1` (comb): PASS  **PASS, passed**
- `Prob024_hadd` (comb): PASS  **PASS, passed**
- `Prob029_m2014_q4g` (comb): PASS  **PASS, passed**
- `Prob031_dff` (seq): X → X → X → X → X → X  **FAIL, budget**
- `Prob033_ece241_2014_q1c` (comb): PASS  **PASS, passed**
- `Prob034_dff8` (seq): R1 → X → X → X → X → X  **FAIL, budget**
- `Prob037_review2015_count1k` (seq): PASS  **PASS, passed**
- `Prob038_count15` (seq): PASS  **PASS, passed**
- `Prob041_dff8r` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob044_vectorgates` (comb): comb? → comb? → comb? → comb? → comb? → comb?  **FAIL, budget**
- `Prob051_gates4` (comb): PASS  **PASS, passed**
- `Prob058_alwaysblock2` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob061_2014_q4a` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob063_review2015_shiftcount` (seq): X → X → X → X → X → X  **FAIL, budget**
- `Prob064_vector3` (comb): X → X → X → X → X → X  **FAIL, budget**
- `Prob070_ece241_2013_q2` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob075_counter_2bc` (seq): PASS  **PASS, passed**
- `Prob076_always_case` (comb): PASS  **PASS, passed**
- `Prob079_fsm3onehot` (comb): R1 → comb? → R6 → R1 → R6 → R1  **FAIL, budget**
- `Prob080_timer` (seq): seq@later → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob081_7458` (comb): PASS  **PASS, passed**
- `Prob083_mt2015_q4b` (comb): R8 → R8 → R8 → R8 → R8 → R8  **FAIL, budget**
- `Prob090_circuit1` (comb): PASS  **PASS, passed**
- `Prob102_circuit3` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob105_rotate100` (seq): seq@later → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob107_fsm1s` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob110_fsm2` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob119_fsm3` (seq): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob121_2014_q3bfsm` (seq): R4 → R4 → R4 → R4 → R4 → R4  **FAIL, budget**
- `Prob125_kmap3` (comb): comb? → comb? → comb? → comb? → comb? → comb?  **FAIL, budget**
- `Prob130_circuit5` (comb): R1 → R1 → R1 → R1 → R1 → R1  **FAIL, budget**
- `Prob132_always_if2` (comb): PASS  **PASS, passed**
- `Prob133_2014_q3fsm` (seq): compile: other → compile: other → compile: other → compile: other → compile: other → compile: other  **FAIL, budget**
- `Prob135_m2014_q6b` (comb): comb? → comb? → comb? → R1 → comb? → R1  **FAIL, budget**
- `Prob144_conwaylife` (seq): syntax → syntax → syntax → syntax → compile: other → syntax  **FAIL, budget**
- `Prob145_circuit8` (comb): X → X → X → X → X → X  **FAIL, budget**
- `Prob149_ece241_2013_q4` (seq): seq@later → reset → reset → reset → reset → reset  **FAIL, budget**
- `Prob152_lemmings3` (seq): seq@later → R4 → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**

_Generated by `taxonomy.py` from 19 run file(s). Counts only; no model was used._
