"""plc_checker.py - the deterministic gate for the two-agent IEC 61131-3 loop.

The gate is the ONLY component allowed to say PASS. No LLM ever grants a pass.

It grades a program in two layers:
  1. IronPLC (`ironplcc check`): is this valid IEC 61131-3?
  2. Spec checks: does the program contain what the spec asked for, using the
     exact identifiers the spec dictated?

Because the spec dictates identifiers ("declare EStop : BOOL"), every check is
a precise structural test and every failure maps to a precise instruction.

    python plc_checker.py --selftest          # prove the checker before trusting a score
    python plc_checker.py --selftest --no-compiler   # checker logic only
"""
import argparse
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

I, S = re.I, re.S

# =====================================================================
# Diversity dimensions
# =====================================================================
SECTORS = ["Water Treatment", "Wastewater Treatment", "Oil Refinery", "Oil and Gas Pipeline",
           "Chemical Processing", "Power Generation", "Electrical Substation", "Smart Grid",
           "Renewable Energy", "Manufacturing", "Food Processing", "Pharmaceutical Production",
           "Mining", "HVAC", "Building Automation", "Marine Systems", "Railway Signaling",
           "Supply Chain & Logistics", "Airport Baggage Handling", "Nuclear Plant & Reactors",
           "Agriculture"]
CONTROL_STRATEGIES = ["State Machine", "Sequential Control", "Interlock Logic", "PID Loop",
                      "Batch Processing", "Lead Lag Control", "Pump Alternation", "Cascade Control"]
IEC_FEATURES = ["CASE Statement", "FOR Loop", "WHILE Loop", "Function Block",
                "Array Processing", "Structures"]
SAFETY_FEATURES = ["Emergency Stop", "Safety Shutdown", "SIL Logic", "Redundant Sensors",
                   "Motor Protection", "Watchdog Monitoring"]
OPERATING_MODES = ["Automatic", "Manual", "Maintenance"]
FAILURE_SCENARIOS = ["Sensor Failure", "Motor Overload", "Valve Stuck", "Power Loss",
                     "Tank Overflow", "High Pressure", "Low Pressure"]
SECURITY_PROFILES = ["Secure", "Insecure"]

# Which compiler capabilities each dimension value needs. If the installed
# ironplcc cannot compile a capability probe, values depending on it are never
# sampled, so the model is never given an impossible spec.
CAPABILITY_DEPS = {
    "CASE Statement": {"case"}, "FOR Loop": {"for"}, "WHILE Loop": {"while"},
    "Function Block": {"fb"}, "Array Processing": {"array", "for"}, "Structures": {"struct"},
    "State Machine": {"case"}, "Sequential Control": {"case"}, "Batch Processing": {"case"},
    "Watchdog Monitoring": {"ton"},
}

# =====================================================================
# Text helpers (IEC 61131-3 is case-insensitive, so every regex uses re.I)
# =====================================================================
INT_TYPES = ("INT", "DINT", "UINT", "SINT", "USINT", "UDINT", "LINT", "ULINT")
BLOCK_COMMENT_RE = re.compile(r"\(\*.*?\*\)", S)
LINE_COMMENT_RE = re.compile(r"//[^\n]*")
VAR_BLOCK_RE = re.compile(
    r"\bVAR(?:_INPUT|_OUTPUT|_IN_OUT|_TEMP|_GLOBAL|_EXTERNAL)?\b(?:\s+(?:CONSTANT|RETAIN))?(.*?)\bEND_VAR\b", S | I)
ASSIGN_RE = re.compile(r"\b(\w+(?:\.\w+)?)\s*:=\s*([^;]*);", S)
COND_RE = re.compile(r"\b(?:IF|ELSIF|WHILE)\b(.*?)\b(?:THEN|DO)\b", S | I)
IF_START_RE = re.compile(r"\b(?:IF|ELSIF)\b", I)
BRANCH_RE = re.compile(r"(.*?)\bTHEN\b(.*?)(?=\bELSIF\b|\bELSE\b|\bEND_IF\b)", S | I)
CASE_RE = re.compile(r"\bCASE\s+(\w+)\s+OF\b(.*?)\bEND_CASE\b", S | I)
CASE_LABEL_RE = re.compile(r"(?:^|;)\s*(\d+(?:\s*(?:,|\.\.)\s*\d+)*)\s*:(?!=)", re.M)
FOR_RE = re.compile(r"\bFOR\s+(\w+)\s*:=(.*?)\bTO\b(.*?)\bDO\b(.*?)\bEND_FOR\b", S | I)
WHILE_RE = re.compile(r"\bWHILE\b(.*?)\bDO\b(.*?)\bEND_WHILE\b", S | I)
FB_RE = re.compile(r"\bFUNCTION_BLOCK\s+(\w+)(.*?)\bEND_FUNCTION_BLOCK\b", S | I)
STRUCT_RE = re.compile(r"\bTYPE\s+(\w+)\s*:\s*STRUCT\b(.*?)\bEND_STRUCT\b", S | I)
PROGRAM_RE = re.compile(r"\bPROGRAM\s+\w+\b(.*?)\bEND_PROGRAM\b", S | I)


def strip_comments(code):
    return LINE_COMMENT_RE.sub(" ", BLOCK_COMMENT_RE.sub(" ", code))


def word(name):
    return re.compile(rf"\b{re.escape(name)}\b", I)


def program_text(c):
    m = PROGRAM_RE.search(c)
    return m.group(1) if m else ""


def declarations(c):
    """{lower-case name: upper-case type} from every VAR... END_VAR block."""
    decls = {}
    for block in VAR_BLOCK_RE.findall(c):
        for stmt in block.split(";"):
            if ":" not in stmt:
                continue
            names, _, typ = stmt.partition(":")
            typ = re.sub(r"\s+", " ", typ.split(":=")[0]).strip().upper()
            for n in names.split(","):
                n = n.strip()
                if re.fullmatch(r"[A-Za-z_]\w*", n):
                    decls[n.lower()] = typ
    return decls


def declared(c, name, typ=None):
    t = declarations(c).get(name.lower())
    if t is None:
        return False
    if typ is None:
        return True
    if typ == "INT":
        return t in INT_TYPES
    return t == typ.upper()


def all_declared(c, names, typ):
    return all(declared(c, n, typ) for n in names)


def assignments(c, name):
    return [rhs for lhs, rhs in ASSIGN_RE.findall(c) if lhs.lower() == name.lower()]


def assigned_to(c, name, pattern):
    return any(re.search(pattern, rhs, I | S) for rhs in assignments(c, name))


def in_condition(c, name):
    return any(word(name).search(cond) for cond in COND_RE.findall(c))


def uses(c, name):
    """Read somewhere: in a condition or on the right-hand side of another assignment."""
    if in_condition(c, name):
        return True
    return any(lhs.lower() != name.lower() and word(name).search(rhs) for lhs, rhs in ASSIGN_RE.findall(c))


def if_branches(c):
    """Every (condition, body) branch, nested ones included."""
    out = []
    for m in IF_START_RE.finditer(c):
        b = BRANCH_RE.match(c, m.end())
        if b:
            out.append((b.group(1), b.group(2)))
    return out


def forces_false(c, name, negated=False):
    """True if some branch whose condition tests `name` (or NOT name) assigns FALSE,
    or an assignment gates on it inline (X := ... AND NOT name)."""
    neg = re.compile(rf"\bNOT\s*\(?\s*{re.escape(name)}\b", I)
    for cond, body in if_branches(c):
        hit = neg.search(cond) if negated else (word(name).search(cond) and not neg.search(cond))
        if hit and re.search(r":=\s*FALSE\b", body, I):
            return True
    inline = rf":=[^;]*\bAND\s+{re.escape(name)}\b" if negated else rf":=[^;]*\bAND\s+NOT\s*\(?\s*{re.escape(name)}\b"
    return bool(re.search(inline, c, I))


def case_label_counts(c):
    counts = {}
    for sel, body in CASE_RE.findall(c):
        n = len(set(re.sub(r"\s+", "", lab) for lab in CASE_LABEL_RE.findall(body)))
        counts[sel.lower()] = max(n, counts.get(sel.lower(), 0))
    return counts


def case_on(c, selector, n):
    return declared(c, selector, "INT") and case_label_counts(c).get(selector.lower(), 0) >= n \
        and bool(assignments(c, selector))


def clamps(c, var=None):
    if var:
        if assigned_to(c, var, r"^\s*(LIMIT|MIN|MAX)\s*\("):
            return True
        return bool(re.search(rf"\bIF\s+{re.escape(var)}\s*[<>]=?[^;]*?\bTHEN\s+{re.escape(var)}\s*:=", c, I))
    if re.search(r"\b(LIMIT|MIN|MAX)\s*\(", c, I):
        return True
    return bool(re.search(r"\bIF\s+(\w+)\s*[<>]=?[^;]*?\bTHEN\s+\1\s*:=", c, I))


# =====================================================================
# Individual structural checks
# =====================================================================
def chk_structure(c):
    return len(re.findall(r"\bPROGRAM\s+\w+", c, I)) == 1 and bool(re.search(r"\bEND_PROGRAM\b", c, I))


def chk_for(c):
    d = declarations(c)
    return any(d.get(v.lower(), "") in INT_TYPES for v, *_ in FOR_RE.findall(c))


def chk_while(c):
    for cond, body in WHILE_RE.findall(c):
        for v in re.findall(r"(\w+)\s*<=?\s*[\w.]+", cond):
            if re.search(rf"\b{re.escape(v)}\s*:=\s*{re.escape(v)}\s*\+\s*\d+", body, I):
                return True
    return False


def chk_fb(c):
    fbs = {n.upper() for n, _ in FB_RE.findall(c)}
    p = program_text(c)
    return any(t in fbs and re.search(rf"\b{re.escape(n)}\s*\(", p, I) for n, t in declarations(p).items())


def chk_array(c):
    arrs = [n for n, t in declarations(c).items() if t.startswith("ARRAY")]
    return any(re.search(rf"\b{re.escape(a)}\s*\[", body, I) for *_, body in FOR_RE.findall(c) for a in arrs)


def chk_struct(c):
    types = {n.upper() for n, body in STRUCT_RE.findall(c) if body.count(";") >= 2}
    p = program_text(c)
    return any(t in types and re.search(rf"\b{re.escape(n)}\.\w+", p, I) for n, t in declarations(p).items())


def chk_pid(c):
    names = ["Setpoint", "ProcessValue", "Kp", "Ki", "Kd", "Error", "PrevError", "Integral", "ControlOutput"]
    return (all_declared(c, names, "REAL")
            and assigned_to(c, "Error", r"^\s*\(?\s*[\w.]+\s*-\s*ProcessValue\b")
            and assigned_to(c, "Integral", r"\bIntegral\b")
            and clamps(c, "ControlOutput"))


def chk_cascade(c):
    names = ["OuterSetpoint", "OuterPV", "InnerSetpoint", "InnerPV", "InnerOutput"]
    return (all_declared(c, names, "REAL")
            and assigned_to(c, "InnerSetpoint", r"\b(OuterSetpoint|OuterPV)\b")
            and assigned_to(c, "InnerOutput", r"\bInnerSetpoint\s*-\s*InnerPV\b"))


def chk_shutdown(c):
    reset_clears = any(word("ResetCmd").search(cond) and re.search(r"\bShutdownActive\s*:=\s*FALSE\b", body, I)
                       for cond, body in if_branches(c))
    return (all_declared(c, ["ShutdownActive", "ResetCmd"], "BOOL")
            and bool(re.search(r"\bShutdownActive\s*:=\s*TRUE\b", c, I))
            and reset_clears and forces_false(c, "ShutdownActive"))


def chk_vote(c):
    return (all_declared(c, ["TripA", "TripB", "TripC", "VoteTrip"], "BOOL")
            and any(all(word(t).search(rhs) for t in ("TripA", "TripB", "TripC"))
                    and re.search(r"\bAND\b", rhs, I) and re.search(r"\bOR\b", rhs, I)
                    for rhs in assignments(c, "VoteTrip"))
            and forces_false(c, "VoteTrip"))


def chk_redundant(c):
    if not (all_declared(c, ["SensorA", "SensorB", "MaxDeviation"], "REAL") and declared(c, "SensorMismatch", "BOOL")):
        return False
    for rhs in assignments(c, "SensorMismatch"):
        if not word("MaxDeviation").search(rhs):
            continue
        if re.search(r"\bABS\s*\(\s*Sensor[AB]\s*-\s*Sensor[AB]\s*\)", rhs, I):
            return True
        if re.search(r"SensorA\s*-\s*SensorB", rhs, I) and re.search(r"SensorB\s*-\s*SensorA", rhs, I):
            return True
    return False


def chk_watchdog(c):
    return (declared(c, "WatchdogTimer", "TON") and declared(c, "Heartbeat", "BOOL")
            and bool(re.search(r"\bWatchdogTimer\s*\(", c, I))
            and (in_condition(c, "WatchdogTimer.Q") or uses(c, "WatchdogTimer.Q")))


def chk_threshold(c, alarm, op, limit):
    rev = {">": "<", "<": ">"}[op]
    pat = rf"\bPressure\s*{op}=?\s*{limit}\b|\b{limit}\s*{rev}=?\s*Pressure\b"
    return (all_declared(c, ["Pressure", limit], "REAL") and declared(c, alarm, "BOOL")
            and assigned_to(c, alarm, pat) and uses(c, alarm))


# =====================================================================
# Diagnosers: when a check fails, say exactly why, with the program's own names
# =====================================================================
def _fb_inputs(c, fb):
    for name, body in FB_RE.findall(c):
        if name.upper() == fb.upper():
            m = re.search(r"\bVAR_INPUT\b(.*?)\bEND_VAR\b", body, S | I)
            if m:
                return [n.strip() for st in m.group(1).split(";") if ":" in st
                        for n in st.split(":")[0].split(",") if n.strip()]
    return []


def diag_fb(c):
    fbs = [n for n, _ in FB_RE.findall(c)]
    stripped = strip_comments(c)
    if re.search(r"\bPROGRAM\b.*\bVAR_INPUT\b", stripped, S | I):
        return ("The FUNCTION_BLOCK definition must come BEFORE PROGRAM Main. The file must look like: "
                "FUNCTION_BLOCK <Name>\nVAR_INPUT\n    <inputs>\nEND_VAR\nVAR_OUTPUT\n    <outputs>\n"
                "END_VAR\n<statements>\nEND_FUNCTION_BLOCK\n\nPROGRAM Main\n...")
    if not fbs:
        return ("No FUNCTION_BLOCK is defined. Before PROGRAM Main, add FUNCTION_BLOCK <Name> with a "
                "VAR_INPUT and a VAR_OUTPUT block, its statements, and END_FUNCTION_BLOCK.")
    p = program_text(c)
    inst = [(n, t) for n, t in declarations(p).items() if t in {f.upper() for f in fbs}]
    if not inst:
        fb = fbs[0]
        return (f"FUNCTION_BLOCK {fb} is defined but PROGRAM Main declares no instance of it. Add "
                f"'{fb}1 : {fb};' to the program's VAR block and call it with {fb}1(...);.")
    name, typ = inst[0]
    real = re.search(rf"\b({re.escape(name)})\b", p, I)
    shown = real.group(1) if real else name
    fb = next(f for f in fbs if f.upper() == typ)
    ins = _fb_inputs(c, fb)
    args = ", ".join(f"{i} := <value>" for i in ins[:3]) or "<input> := <value>"
    extra = (f" Assigning {shown}.{ins[0]} is not a call; pass inputs inside the call instead."
             if ins and re.search(rf"\b{re.escape(name)}\.{re.escape(ins[0])}\s*:=", p, I) else "")
    return (f"'{shown}' is declared as {fb} but never called, so the block never executes.{extra} "
            f"Add the statement {shown}({args}); in the program body and read outputs as {shown}.<output>.")


def diag_struct(c):
    types = STRUCT_RE.findall(c)
    if not types:
        return ("No STRUCT type is defined. Before PROGRAM Main, add TYPE <Name> : STRUCT <field> : REAL; "
                "<field2> : REAL; END_STRUCT; END_TYPE.")
    name, body = types[0]
    if body.count(";") < 2:
        return f"STRUCT {name} needs at least two fields, each ending with a semicolon."
    p = program_text(c)
    var = next((n for n, t in declarations(p).items() if t == name.upper()), None)
    if not var:
        return f"TYPE {name} is defined but PROGRAM Main declares no variable of it. Add '{name}Data : {name};' to VAR."
    return f"'{var}' is declared as {name} but its fields are never used. Read or write them as {var}.<field>."


def diag_vote(c):
    rhs = assignments(c, "VoteTrip")
    if not rhs:
        return "VoteTrip is never assigned. Add VoteTrip := (TripA AND TripB) OR (TripA AND TripC) OR (TripB AND TripC);."
    if not forces_false(c, "VoteTrip"):
        return "VoteTrip is computed but nothing acts on it. Add IF VoteTrip THEN <each output> := FALSE; END_IF;."
    missing = [n for n in ("TripA", "TripB", "TripC", "VoteTrip") if not declared(c, n, "BOOL")]
    if missing:
        return f"Declare {', '.join(missing)} as BOOL in the VAR block."
    return "VoteTrip must be exactly (TripA AND TripB) OR (TripA AND TripC) OR (TripB AND TripC)."


def diag_declared_forced(name, outputs_hint, negated=False):
    def d(c):
        if not declared(c, name, "BOOL"):
            return f"Declare {name} : BOOL in the VAR block."
        test = f"NOT {name}" if negated else name
        return f"Nothing forces outputs off when {test}. Add IF {test} THEN {outputs_hint} := FALSE; END_IF; at the end of the program."
    return d


# =====================================================================
# Requirement library: dimension value -> requirements
# =====================================================================
@dataclass
class Req:
    id: str
    text: str                  # shown to the generator
    hint: str                  # deterministic fix instruction when the check fails
    check: object              # callable(str) -> bool
    raw: bool = False          # True: check sees comments (needed for the VULN marker)
    diagnose: object = None    # callable(str) -> str: a fix naming the program's own identifiers


STRUCTURE = Req("structure", "Exactly one PROGRAM Main ... END_PROGRAM.",
                "The file must contain exactly one PROGRAM Main and end it with END_PROGRAM.", chk_structure)

MOTOR = Req("motor_overload",
            "Declare MotorOverload : BOOL and MotorRun : BOOL. Inside IF MotorOverload THEN set MotorRun := FALSE;.",
            "Declare MotorOverload : BOOL and MotorRun : BOOL, and add IF MotorOverload THEN MotorRun := FALSE; END_IF;.",
            lambda c: all_declared(c, ["MotorOverload", "MotorRun"], "BOOL") and forces_false(c, "MotorOverload"))

REQS = {
    # ---- IEC language features ----
    "CASE Statement": [Req("case_statement",
        "Use at least one CASE <INT variable> OF ... END_CASE; with two or more numbered branches (0:, 1:).",
        "Add a CASE block on an INT variable with numbered branches such as 0: and 1:, closed with END_CASE;.",
        lambda c: any(n >= 2 for n in case_label_counts(c).values()))],
    "FOR Loop": [Req("for_loop",
        "Use a FOR i := a TO b DO ... END_FOR; loop whose counter is declared as INT.",
        "Add a FOR loop whose counter variable is declared as INT, closed with END_FOR;.", chk_for)],
    "WHILE Loop": [Req("while_loop",
        "Use a WHILE ... DO ... END_WHILE; loop that must terminate: its condition compares a counter "
        "with < or <= to a limit, and the body increments that counter (n := n + 1;).",
        "Make the WHILE condition compare a counter with < or <= and increment it inside the loop with n := n + 1;.",
        chk_while)],
    "Function Block": [Req("function_block",
        "Before the PROGRAM, define a FUNCTION_BLOCK with VAR_INPUT and VAR_OUTPUT. In the PROGRAM's VAR block, "
        "declare an instance of it (for example Ctl1 : MyBlock;) and call that instance every scan using the "
        "same name (Ctl1(In1 := x);).",
        "Define a FUNCTION_BLOCK before PROGRAM Main, declare an instance of it in the program's VAR block, "
        "and call that instance by its declared name every scan.", chk_fb, diagnose=diag_fb)],
    "Array Processing": [Req("array_processing",
        "Declare an ARRAY [lo..hi] OF REAL (or INT) and read or write it by index inside a FOR loop.",
        "Declare a variable as ARRAY [1..N] OF REAL and index it, Arr[i], inside a FOR loop body.", chk_array)],
    "Structures": [Req("structures",
        "Before the PROGRAM, define TYPE Name : STRUCT ... END_STRUCT; END_TYPE with at least two fields, "
        "declare a variable of that type in the PROGRAM, and use its fields with dot notation (Var.Field).",
        "Define TYPE X : STRUCT with two or more fields END_STRUCT; END_TYPE before PROGRAM Main, declare a "
        "variable of type X in the program, and read or write Var.Field.", chk_struct, diagnose=diag_struct)],

    # ---- control strategies ----
    "State Machine": [Req("state_machine",
        "Declare State : INT and drive the logic with CASE State OF having at least 3 numbered states "
        "(0:, 1:, 2:); assign State inside branches to move between states.",
        "Declare State : INT and write CASE State OF with branches 0:, 1:, 2: that assign State.",
        lambda c: case_on(c, "State", 3))],
    "Sequential Control": [Req("sequential_control",
        "Declare SeqStep : INT and run the sequence with CASE SeqStep OF having at least 3 numbered steps; "
        "advance SeqStep only when the current step's completion condition is TRUE.",
        "Declare SeqStep : INT and write CASE SeqStep OF with steps 0:, 1:, 2: that assign SeqStep.",
        lambda c: case_on(c, "SeqStep", 3))],
    "Batch Processing": [Req("batch_processing",
        "Declare Phase : INT and BatchComplete : BOOL. Run CASE Phase OF with at least 3 numbered phases "
        "(e.g. fill, react, drain) and set BatchComplete := TRUE; in the final phase.",
        "Declare Phase : INT and BatchComplete : BOOL, write CASE Phase OF with 3+ numbered phases, "
        "and assign BatchComplete := TRUE; in the last one.",
        lambda c: case_on(c, "Phase", 3) and declared(c, "BatchComplete", "BOOL")
        and assigned_to(c, "BatchComplete", r"\bTRUE\b"))],
    "Interlock Logic": [Req("interlock",
        "Declare Permissive : BOOL, assign it from at least two conditions joined with AND, and let every "
        "actuator output be TRUE only when Permissive is TRUE.",
        "Declare Permissive : BOOL, assign Permissive := cond1 AND cond2;, and use Permissive in the "
        "conditions or assignments that drive the outputs.",
        lambda c: declared(c, "Permissive", "BOOL") and assigned_to(c, "Permissive", r"\bAND\b")
        and uses(c, "Permissive"))],
    "PID Loop": [Req("pid_loop",
        "Declare REAL Setpoint, ProcessValue, Kp, Ki, Kd, Error, PrevError, Integral, ControlOutput. "
        "Compute Error := Setpoint - ProcessValue;, accumulate Integral := Integral + Ki * Error;, and clamp "
        "ControlOutput to 0.0..100.0 with ControlOutput := LIMIT(0.0, ..., 100.0); or an IF clamp.",
        "Declare all nine PID variables as REAL, assign Error := Setpoint - ProcessValue;, update Integral from "
        "itself, and clamp ControlOutput with LIMIT(0.0, value, 100.0).", chk_pid)],
    "Lead Lag Control": [Req("lead_lag",
        "Declare LeadPump : BOOL and LagPump : BOOL. Start LeadPump on demand; start LagPump only when demand "
        "exceeds a second, higher threshold.",
        "Declare LeadPump and LagPump as BOOL and assign both, LagPump using a higher demand threshold.",
        lambda c: all_declared(c, ["LeadPump", "LagPump"], "BOOL") and bool(assignments(c, "LeadPump"))
        and bool(assignments(c, "LagPump")))],
    "Pump Alternation": [Req("pump_alternation",
        "Declare BOOL PumpA, PumpB and LeadIsA. Toggle duty with LeadIsA := NOT LeadIsA; at the end of each "
        "pump cycle; run PumpA when LeadIsA is TRUE, otherwise PumpB.",
        "Declare PumpA, PumpB, LeadIsA as BOOL, toggle with LeadIsA := NOT LeadIsA;, and assign PumpA and "
        "PumpB from LeadIsA.",
        lambda c: all_declared(c, ["PumpA", "PumpB", "LeadIsA"], "BOOL")
        and assigned_to(c, "LeadIsA", r"^\s*NOT\s+LeadIsA\s*$") and bool(assignments(c, "PumpA"))
        and bool(assignments(c, "PumpB")))],
    "Cascade Control": [Req("cascade",
        "Declare REAL OuterSetpoint, OuterPV, InnerSetpoint, InnerPV, InnerOutput. The outer loop writes "
        "InnerSetpoint from OuterSetpoint and OuterPV; the inner loop computes InnerOutput from "
        "InnerSetpoint - InnerPV.",
        "Declare the five cascade REALs, assign InnerSetpoint from an expression using OuterSetpoint/OuterPV, "
        "and assign InnerOutput from an expression containing InnerSetpoint - InnerPV.", chk_cascade)],

    # ---- safety ----
    "Emergency Stop": [Req("emergency_stop",
        "Declare EStop : BOOL (TRUE = stop pressed). Inside IF EStop THEN set every actuator output to FALSE.",
        "Declare EStop : BOOL and add IF EStop THEN <each output> := FALSE; END_IF; at the end of the program.",
        lambda c: declared(c, "EStop", "BOOL") and forces_false(c, "EStop"),
        diagnose=diag_declared_forced("EStop", "<each actuator output>"))],
    "Safety Shutdown": [Req("safety_shutdown",
        "Declare BOOL ShutdownActive and ResetCmd. Latch ShutdownActive := TRUE; when a trip occurs, clear it "
        "only inside an IF whose condition includes ResetCmd, and while ShutdownActive is TRUE force all "
        "actuator outputs FALSE.",
        "Declare ShutdownActive and ResetCmd as BOOL, latch ShutdownActive := TRUE; on a trip, clear it in "
        "IF ResetCmd ... THEN ShutdownActive := FALSE;, and add IF ShutdownActive THEN outputs := FALSE;.",
        chk_shutdown)],
    "SIL Logic": [Req("sil_voting",
        "Declare BOOL TripA, TripB, TripC, VoteTrip. Implement 2-out-of-3 voting: "
        "VoteTrip := (TripA AND TripB) OR (TripA AND TripC) OR (TripB AND TripC); and force actuator outputs "
        "FALSE when VoteTrip is TRUE.",
        "Assign VoteTrip := (TripA AND TripB) OR (TripA AND TripC) OR (TripB AND TripC); and add "
        "IF VoteTrip THEN outputs := FALSE; END_IF;.", chk_vote, diagnose=diag_vote)],
    "Redundant Sensors": [Req("redundant_sensors",
        "Declare REAL SensorA, SensorB, MaxDeviation and BOOL SensorMismatch. "
        "Set SensorMismatch := ABS(SensorA - SensorB) > MaxDeviation;.",
        "Declare SensorA, SensorB, MaxDeviation as REAL and SensorMismatch as BOOL, and assign "
        "SensorMismatch := ABS(SensorA - SensorB) > MaxDeviation;.", chk_redundant)],
    "Motor Protection": [MOTOR],
    "Watchdog Monitoring": [Req("watchdog",
        "Declare WatchdogTimer : TON and Heartbeat : BOOL. Call WatchdogTimer(IN := NOT Heartbeat, PT := T#2s); "
        "every scan, and when WatchdogTimer.Q is TRUE force actuator outputs FALSE.",
        "Declare WatchdogTimer : TON, call WatchdogTimer(IN := NOT Heartbeat, PT := T#2s); each scan, and "
        "test WatchdogTimer.Q in an IF that sets outputs FALSE.", chk_watchdog)],

    # ---- failure scenarios ----
    "Sensor Failure": [Req("sensor_failure",
        "Declare SensorFault : BOOL and set it from a range check on a raw REAL input, e.g. "
        "SensorFault := (RawValue < 0.0) OR (RawValue > 100.0);, then use it to reach a safe state.",
        "Declare SensorFault : BOOL, assign it from a < / > range check, and use SensorFault in an IF or "
        "interlock.",
        lambda c: declared(c, "SensorFault", "BOOL") and assigned_to(c, "SensorFault", r"[<>]")
        and uses(c, "SensorFault"))],
    "Motor Overload": [MOTOR],
    "Valve Stuck": [Req("valve_stuck",
        "Declare BOOL ValveCmd, ValveOpenFb, ValveStuck. Set ValveStuck when ValveCmd and ValveOpenFb disagree "
        "(XOR or <>), and alarm or go safe when ValveStuck is TRUE.",
        "Declare ValveCmd, ValveOpenFb, ValveStuck as BOOL, assign ValveStuck := ValveCmd XOR ValveOpenFb;, "
        "and use ValveStuck in an IF.",
        lambda c: all_declared(c, ["ValveCmd", "ValveOpenFb", "ValveStuck"], "BOOL")
        and assigned_to(c, "ValveStuck", r"(?=.*\bValveCmd\b)(?=.*\bValveOpenFb\b)") and uses(c, "ValveStuck"))],
    "Power Loss": [Req("power_loss",
        "Declare PowerOK : BOOL. Inside IF NOT PowerOK THEN force every actuator output FALSE.",
        "Declare PowerOK : BOOL and add IF NOT PowerOK THEN <each output> := FALSE; END_IF;.",
        lambda c: declared(c, "PowerOK", "BOOL") and forces_false(c, "PowerOK", negated=True),
        diagnose=diag_declared_forced("PowerOK", "<each actuator output>", negated=True))],
    "Tank Overflow": [Req("tank_overflow",
        "Declare LevelHighHigh : BOOL and InletValve : BOOL. Inside IF LevelHighHigh THEN set InletValve := FALSE;.",
        "Declare LevelHighHigh and InletValve as BOOL and add IF LevelHighHigh THEN InletValve := FALSE; END_IF;.",
        lambda c: all_declared(c, ["LevelHighHigh", "InletValve"], "BOOL") and forces_false(c, "LevelHighHigh"))],
    "High Pressure": [Req("high_pressure",
        "Declare REAL Pressure, HighPressureLimit and BOOL HighPressureAlarm. "
        "Set HighPressureAlarm := Pressure > HighPressureLimit; and act on it.",
        "Assign HighPressureAlarm := Pressure > HighPressureLimit; (REAL inputs, BOOL alarm) and use "
        "HighPressureAlarm in an IF or interlock.",
        lambda c: chk_threshold(c, "HighPressureAlarm", ">", "HighPressureLimit"))],
    "Low Pressure": [Req("low_pressure",
        "Declare REAL Pressure, LowPressureLimit and BOOL LowPressureAlarm. "
        "Set LowPressureAlarm := Pressure < LowPressureLimit; and act on it.",
        "Assign LowPressureAlarm := Pressure < LowPressureLimit; (REAL inputs, BOOL alarm) and use "
        "LowPressureAlarm in an IF or interlock.",
        lambda c: chk_threshold(c, "LowPressureAlarm", "<", "LowPressureLimit"))],

    # ---- operating mode ----
    "Automatic": [Req("mode_auto",
        "Declare AutoMode : BOOL; normal control logic runs inside IF AutoMode THEN ... END_IF;.",
        "Declare AutoMode : BOOL and wrap the normal control logic in IF AutoMode THEN ... END_IF;.",
        lambda c: declared(c, "AutoMode", "BOOL") and in_condition(c, "AutoMode"))],
    "Manual": [Req("mode_manual",
        "Declare ManualMode : BOOL; inside IF ManualMode THEN outputs follow manual command BOOLs, "
        "and safety conditions still override them.",
        "Declare ManualMode : BOOL and add an IF ManualMode THEN branch where outputs follow manual commands.",
        lambda c: declared(c, "ManualMode", "BOOL") and in_condition(c, "ManualMode"))],
    "Maintenance": [Req("mode_maintenance",
        "Declare MaintenanceMode : BOOL; inside IF MaintenanceMode THEN actuators stay off unless a jog command "
        "is held, and safety conditions still override.",
        "Declare MaintenanceMode : BOOL and add an IF MaintenanceMode THEN branch that holds actuators off.",
        lambda c: declared(c, "MaintenanceMode", "BOOL") and in_condition(c, "MaintenanceMode"))],

    # ---- security profile ----
    "Secure": [Req("secure_inputs",
        "Validate every REAL setpoint input before use: clamp it with LIMIT(min, value, max) or an IF range clamp.",
        "Clamp each REAL setpoint input before use, e.g. SafeSP := LIMIT(0.0, Setpoint, 100.0);.",
        clamps)],
    "Insecure": [Req("vuln_marker",
        "Include exactly one realistic PLC weakness (for example an unvalidated setpoint or a maintenance bypass "
        "that skips an interlock) and mark it with a comment (* VULN: short description *).",
        "Add one realistic weakness and mark it with a comment that starts (* VULN: .",
        lambda raw: bool(re.search(r"\(\*\s*VULN:", raw, I)), raw=True)],
}


@dataclass
class Spec:
    metadata: dict
    reqs: list = field(default_factory=list)


def reqs_for(values):
    out, seen = [STRUCTURE], {"structure"}
    for v in values:
        for r in REQS[v]:
            if r.id not in seen:
                out.append(r)
                seen.add(r.id)
    return out


# Curriculum levels: each adds one kind of requirement. Level 4 is the full random sampling.
#   1  mode + one simple safety interlock
#   2  + a control strategy, + one more safety feature
#   3  + one IEC language feature, + one failure scenario, + a security profile
#   4  full spec: 1-2 IEC features, 1-2 safety features, 1-2 failure scenarios
EASY_SAFETY = ["Emergency Stop", "Motor Protection", "Power Loss", "Tank Overflow"]
MAX_LEVEL = 4


def build_spec(rng, unsupported=frozenset(), level=MAX_LEVEL):
    def ok(vals):
        return [v for v in vals if not (CAPABILITY_DEPS.get(v, set()) & set(unsupported))]
    md = {"sector": rng.choice(SECTORS), "level": level, "control_strategy": None,
          "operating_mode": rng.choice(OPERATING_MODES), "security_profile": None,
          "iec_features": [], "safety_features": [], "failure_scenarios": []}
    feats, safety = ok(IEC_FEATURES), ok(SAFETY_FEATURES)
    if level <= 1:
        easy = rng.choice(EASY_SAFETY)
        md["safety_features" if easy in SAFETY_FEATURES else "failure_scenarios"] = [easy]
    elif level == 2:
        md["control_strategy"] = rng.choice(ok(CONTROL_STRATEGIES))
        md["safety_features"] = [rng.choice(safety)]
        md["failure_scenarios"] = [rng.choice(["Power Loss", "Tank Overflow"])]
    elif level == 3:
        md["control_strategy"] = rng.choice(ok(CONTROL_STRATEGIES))
        md["security_profile"] = rng.choice(SECURITY_PROFILES)
        md["iec_features"] = [rng.choice(feats)]
        md["safety_features"] = [rng.choice(safety)]
        md["failure_scenarios"] = [rng.choice(FAILURE_SCENARIOS)]
    else:
        md["control_strategy"] = rng.choice(ok(CONTROL_STRATEGIES))
        md["security_profile"] = rng.choice(SECURITY_PROFILES)
        md["iec_features"] = rng.sample(feats, min(len(feats), rng.randint(1, 2)))
        md["safety_features"] = rng.sample(safety, rng.randint(1, 2))
        md["failure_scenarios"] = rng.sample(FAILURE_SCENARIOS, rng.randint(1, 2))
    values = ([md["control_strategy"]] if md["control_strategy"] else []) + md["iec_features"] \
        + md["safety_features"] + md["failure_scenarios"] + [md["operating_mode"]] \
        + ([md["security_profile"]] if md["security_profile"] else [])
    reqs = reqs_for(values)
    md["requirements"] = [r.id for r in reqs]
    return Spec(md, reqs)


def spec_prompt(spec):
    md = spec.metadata
    reqs = "\n".join(f"{i}. {r.text}" for i, r in enumerate(spec.reqs, 1))
    control = md['control_strategy'] or 'simple on/off control'
    return f"""Write one Structured Text program for: {md['sector']}, {control}, primary mode {md['operating_mode']}.

Format rules:
- File layout, in this order (include TYPE and FUNCTION_BLOCK parts only if a requirement asks for them):
    TYPE ... END_TYPE
    FUNCTION_BLOCK <Name> VAR_INPUT ... END_VAR VAR_OUTPUT ... END_VAR <statements> END_FUNCTION_BLOCK
    PROGRAM Main
    VAR
        <every program variable>
    END_VAR
    <statements>
    END_PROGRAM
- The line PROGRAM Main comes directly before the program's VAR block. No VAR block may appear outside a PROGRAM or FUNCTION_BLOCK.
- Declare every variable you use, including loop counters, with its type. Use the exact names given below.
- End every statement with a semicolon, including END_IF; END_CASE; END_FOR; END_WHILE;
- Comments only as (* ... *), and keep them short.
- Put safety overrides at the end of the program so they win over normal logic.

Requirements (each is checked automatically):
{reqs}"""


# =====================================================================
# Compiler
# =====================================================================
class CompilerMissing(RuntimeError):
    pass


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def compile_check(code, compiler_cmd="ironplcc check", timeout=60):
    argv = shlex.split(compiler_cmd)
    if shutil.which(argv[0]) is None and not os.path.exists(argv[0]):
        raise CompilerMissing(f"'{argv[0]}' not found. Install IronPLC or pass --compiler. See README.")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "program.st")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            res = subprocess.run(argv + [path], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "Compiler timed out."
    out = ANSI_RE.sub("", (res.stdout or "") + (res.stderr or ""))
    out = out.replace(path, "program.st").replace(d + os.sep, "")
    return res.returncode == 0, out.strip()


# ---- turning IronPLC output into instructions ----
_HEADER_RE = re.compile(r"^(error|warning)\[(P\d+)\]:\s*(.*)$", re.M)
_LOC_RE = re.compile(r"program\.st:(\d+):(\d+)")
_CARET_RE = re.compile(r"^\s*[│|]\s*\^+\s*(.*)$", re.M)
_SRC_RE = re.compile(r"^\s*(\d+)\s*[│|] ?(.*)$", re.M)
_NOISE_TOKENS = {" ", "\\t", "\\n", "\t", "\n", "(* ... *)", "{ ... }"}


def parse_compiler(output):
    """IronPLC diagnostics -> [{code, title, line, col, source, expected, found, args}]"""
    problems = []
    heads = list(_HEADER_RE.finditer(output))
    for k, h in enumerate(heads):
        if h.group(1) != "error":
            continue
        block = output[h.end(): heads[k + 1].start() if k + 1 < len(heads) else len(output)]
        p = {"code": h.group(2), "title": h.group(3).strip(), "line": None, "col": None,
             "source": "", "expected": [], "found": None, "args": {}}
        p["args"] = dict(re.findall(r"(\w+)=([^)\s,]+)", p["title"]))
        p["title"] = re.sub(r"\s*\([^)]*=[^)]*\)\s*$", "", p["title"])
        m = _LOC_RE.search(block)
        if m:
            p["line"], p["col"] = int(m.group(1)), int(m.group(2))
        for num, text in _SRC_RE.findall(block):
            if p["line"] and int(num) == p["line"]:
                p["source"] = text.strip()
        c = _CARET_RE.search(block)
        if c:
            detail = c.group(1)
            exp, _, found = detail.partition("Found")
            p["expected"] = [t for t in re.findall(r"'([^']*)'", exp) if t not in _NOISE_TOKENS]
            f = re.search(r"'([^']*)'", found)
            p["found"] = f.group(1) if f else None
        problems.append(p)
    return problems


def problem_key(p):
    """Position-free identity of a mistake, so the same mistake matches across programs."""
    key = p["code"]
    if p["found"]:
        key += f" found '{p['found']}'"
    if p["args"]:
        key += " " + " ".join(f"{a}" for a in p["args"])
    return key


def hint_for(p):
    """A deterministic instruction for known IronPLC problems."""
    L = f"Line {p['line']}: " if p["line"] else ""
    exp = set(p["expected"])
    if p["code"] == "P0002" and p["found"] and p["found"].upper().startswith("VAR") and "PROGRAM" in exp:
        return (L + f"this {p['found']} block is outside any PROGRAM or FUNCTION_BLOCK. Put the line "
                "'PROGRAM Main' directly above the program's VAR block, and end the file with END_PROGRAM.")
    if p["code"] == "P0002":
        found = f"found '{p['found']}'" if p["found"] else "found an unexpected token"
        shown = ", ".join(sorted(exp)[:8]) or "a different token"
        return (L + f"syntax error at '{p['source'][:60]}': {found}, expected one of {shown}. Usually the "
                "statement just before this line is missing a semicolon, a keyword, or its END_ keyword.")
    if p["code"] == "P4007":
        v = p["args"].get("variable", "this variable")
        return L + f"'{v}' is used but never declared. Add '{v} : <type>;' (e.g. INT for a counter) to the VAR block."
    if p["code"] == "P2008":
        ident = p["args"].get("identifier", "this type")
        return (L + f"'{ident}' is used as a type but is not defined. If it is a FUNCTION_BLOCK name, the "
                f"FUNCTION_BLOCK {ident} ... END_FUNCTION_BLOCK definition must appear BEFORE PROGRAM Main "
                f"in the file, not inside it. Check that the definition exists and is placed before PROGRAM.")
    if p["code"] == "P4012":
        v = p["args"].get("invocation", "this name")
        return (L + f"'{v}(...)' calls a function block instance that is not declared. Declare it in the "
                f"PROGRAM's VAR block as '{v} : <FunctionBlockName>;', or call the instance name you did declare.")
    return L + f"{p['code']} {p['title']}" + (f" at '{p['source'][:60]}'" if p["source"] else "")


def explain_compiler(output, limit=4):
    """Compiler output as numbered instructions; falls back to trimmed raw output if unparsed."""
    problems = parse_compiler(output)
    if not problems:
        return output[:1500]
    lines = [f"{i}. {hint_for(p)}" for i, p in enumerate(problems[:limit], 1)]
    if len(problems) > limit:
        lines.append(f"({len(problems) - limit} more problems; fix these first.)")
    return "\n".join(lines)


# =====================================================================
# Grading
# =====================================================================
@dataclass
class Verdict:
    score: float
    compiled: bool
    compiler_output: str
    passed: list
    failed: list
    truncated: bool = False
    empty: bool = False
    hints: dict = field(default_factory=dict)   # requirement id -> specific fix instruction

    def hint(self, r):
        return self.hints.get(r.id, r.hint)

    @property
    def ok(self):
        return self.compiled and not self.failed and not self.truncated and not self.empty

    @property
    def frac(self):
        total = len(self.passed) + len(self.failed)
        return len(self.passed) / total if total else 0.0


def run_checks(code, reqs, hints=None):
    stripped = strip_comments(code)
    passed, failed = [], []
    for r in reqs:
        text = code if r.raw else stripped
        try:
            ok = bool(r.check(text))
        except Exception:
            ok = False
        (passed if ok else failed).append(r)
        if not ok and hints is not None and r.diagnose:
            try:
                hints[r.id] = r.diagnose(text)
            except Exception:
                pass
    return passed, failed


def grade(code, spec, compiler_cmd="ironplcc check", truncated=False):
    """Score: 0.0 no code / truncated, 0.3 compiler rejects, 0.6-1.0 compiles (0.6 + 0.4 * share of
    requirement checks passed). PASS means compiled and every check passed, i.e. score 1.0."""
    if truncated or not code.strip():
        return Verdict(0.0, False, "", [], list(spec.reqs), truncated=truncated, empty=not code.strip())
    compiled, out = compile_check(code, compiler_cmd)
    hints = {}
    passed, failed = run_checks(code, spec.reqs, hints)
    v = Verdict(0.0, compiled, out, passed, failed, hints=hints)
    v.score = round(0.6 + 0.4 * v.frac, 3) if compiled else 0.3
    return v


# =====================================================================
# Compiler capability probe
# =====================================================================
def _prog(decls, body, pre=""):
    return f"{pre}PROGRAM Main\nVAR\n{decls}\nEND_VAR\n{body}\nEND_PROGRAM\n"


GOOD_PROBES = {
    "minimal": _prog("    x : INT;", "x := x + 1;"),
    "case": _prog("    s : INT;", "CASE s OF\n    0: s := 1;\n    1: s := 0;\nEND_CASE;"),
    "for": _prog("    i : INT;\n    t : INT;", "FOR i := 1 TO 3 DO\n    t := t + i;\nEND_FOR;"),
    "while": _prog("    n : INT;", "n := 0;\nWHILE n < 3 DO\n    n := n + 1;\nEND_WHILE;"),
    "fb": _prog("    inst : Doubler;\n    y : INT;", "inst(X := 2);\ny := inst.Y;",
                pre="FUNCTION_BLOCK Doubler\nVAR_INPUT\n    X : INT;\nEND_VAR\nVAR_OUTPUT\n    Y : INT;\n"
                    "END_VAR\nY := X * 2;\nEND_FUNCTION_BLOCK\n\n"),
    "array": _prog("    a : ARRAY [1..3] OF INT;\n    i : INT;", "FOR i := 1 TO 3 DO\n    a[i] := i;\nEND_FOR;"),
    "struct": _prog("    p : Pair;", "p.A := 1.0;\np.B := p.A;",
                    pre="TYPE Pair :\nSTRUCT\n    A : REAL;\n    B : REAL;\nEND_STRUCT;\nEND_TYPE\n\n"),
    "ton": _prog("    t : TON;\n    go : BOOL;\n    done : BOOL;", "t(IN := go, PT := T#2s);\ndone := t.Q;"),
}
BAD_PROBES = {
    "missing_end_if": _prog("    x : INT;", "IF x > 0 THEN\n    x := 0;"),
    "garbage_token": _prog("    x : INT;", "x := := 1;"),
    "missing_end_program": "PROGRAM Main\nVAR\n    x : INT;\nEND_VAR\nx := 1;\n",
}


def probe_compiler(compiler_cmd="ironplcc check"):
    """Returns (unsupported capabilities, gate_ok, details). gate_ok is False if the compiler accepts
    programs that are obviously invalid, or rejects the minimal valid program."""
    details, unsupported = {}, set()
    for name, code in GOOD_PROBES.items():
        ok, out = compile_check(code, compiler_cmd)
        details[name] = (ok, out)
        if not ok:
            unsupported.add(name)
    gate_ok = "minimal" not in unsupported
    for name, code in BAD_PROBES.items():
        ok, out = compile_check(code, compiler_cmd)
        details["bad:" + name] = (not ok, out)
        if ok:
            gate_ok = False
    return unsupported, gate_ok, details


# =====================================================================
# Selftest
# =====================================================================
KITCHEN_SINK = """TYPE TankData :
STRUCT
    Level : REAL;
    Temperature : REAL;
END_STRUCT;
END_TYPE

FUNCTION_BLOCK Hysteresis
VAR_INPUT
    Value : REAL;
    OnAbove : REAL;
    OffBelow : REAL;
END_VAR
VAR_OUTPUT
    Active : BOOL;
END_VAR
IF Value > OnAbove THEN
    Active := TRUE;
ELSIF Value < OffBelow THEN
    Active := FALSE;
END_IF;
END_FUNCTION_BLOCK

PROGRAM Main
VAR
    AutoMode, ManualMode, MaintenanceMode, ManualPumpCmd : BOOL;
    EStop, ResetCmd, ShutdownActive : BOOL;
    TripA, TripB, TripC, VoteTrip : BOOL;
    SensorA, SensorB, MaxDeviation : REAL;
    SensorMismatch : BOOL;
    MotorOverload, MotorRun, Heartbeat : BOOL;
    WatchdogTimer : TON;
    RawValue : REAL;
    SensorFault : BOOL;
    ValveCmd, ValveOpenFb, ValveStuck, PowerOK : BOOL;
    LevelHighHigh, InletValve : BOOL;
    Pressure, HighPressureLimit, LowPressureLimit : REAL;
    HighPressureAlarm, LowPressureAlarm : BOOL;
    State, SeqStep, Phase : INT;
    BatchComplete, Permissive : BOOL;
    Setpoint, ProcessValue, Kp, Ki, Kd, Error, PrevError, Integral, ControlOutput : REAL;
    LeadPump, LagPump, PumpA, PumpB, LeadIsA, CycleDone : BOOL;
    Demand, SafeSetpoint : REAL;
    OuterSetpoint, OuterPV, InnerSetpoint, InnerPV, InnerOutput : REAL;
    Levels : ARRAY [1..4] OF REAL;
    LevelSum : REAL;
    i, Retries : INT;
    Tank : TankData;
    HighLevel : Hysteresis;
END_VAR

WatchdogTimer(IN := NOT Heartbeat, PT := T#2s);
VoteTrip := (TripA AND TripB) OR (TripA AND TripC) OR (TripB AND TripC);
SensorMismatch := ABS(SensorA - SensorB) > MaxDeviation;
SensorFault := (RawValue < 0.0) OR (RawValue > 100.0);
ValveStuck := ValveCmd XOR ValveOpenFb;
HighPressureAlarm := Pressure > HighPressureLimit;
LowPressureAlarm := Pressure < LowPressureLimit;
(* VULN: HMI setpoint accepted without operator authorization *)
SafeSetpoint := LIMIT(0.0, Setpoint, 100.0);
Permissive := PowerOK AND NOT SensorFault AND NOT SensorMismatch;

IF VoteTrip OR HighPressureAlarm OR LowPressureAlarm OR ValveStuck THEN
    ShutdownActive := TRUE;
END_IF;
IF ResetCmd AND NOT VoteTrip THEN
    ShutdownActive := FALSE;
END_IF;

LevelSum := 0.0;
FOR i := 1 TO 4 DO
    LevelSum := LevelSum + Levels[i];
END_FOR;
Tank.Level := LevelSum / 4.0;
HighLevel(Value := Tank.Level, OnAbove := 90.0, OffBelow := 80.0);

Retries := 0;
WHILE Retries < 3 AND NOT PowerOK DO
    Retries := Retries + 1;
END_WHILE;

IF AutoMode THEN
    CASE State OF
        0: IF Permissive THEN State := 1; END_IF;
        1: InletValve := TRUE;
           IF HighLevel.Active THEN State := 2; END_IF;
        2: InletValve := FALSE;
           State := 0;
    END_CASE;
    CASE SeqStep OF
        0: IF Permissive THEN SeqStep := 1; END_IF;
        1: IF HighLevel.Active THEN SeqStep := 2; END_IF;
        2: SeqStep := 0;
    END_CASE;
    CASE Phase OF
        0: BatchComplete := FALSE;
           Phase := 1;
        1: Phase := 2;
        2: BatchComplete := TRUE;
           Phase := 0;
    END_CASE;
    Error := SafeSetpoint - ProcessValue;
    Integral := Integral + Ki * Error;
    ControlOutput := LIMIT(0.0, Kp * Error + Integral + Kd * (Error - PrevError), 100.0);
    PrevError := Error;
    LeadPump := Demand > 50.0;
    LagPump := Demand > 80.0;
    IF CycleDone THEN
        LeadIsA := NOT LeadIsA;
    END_IF;
    PumpA := LeadIsA AND LeadPump;
    PumpB := (NOT LeadIsA) AND LeadPump;
    InnerSetpoint := LIMIT(0.0, OuterSetpoint - OuterPV, 100.0);
    InnerOutput := InnerSetpoint - InnerPV;
    MotorRun := LeadPump AND Permissive;
ELSIF ManualMode THEN
    MotorRun := ManualPumpCmd AND Permissive;
ELSIF MaintenanceMode THEN
    MotorRun := FALSE;
    InletValve := FALSE;
END_IF;

IF EStop OR ShutdownActive OR VoteTrip OR WatchdogTimer.Q OR MotorOverload OR LevelHighHigh THEN
    MotorRun := FALSE;
    InletValve := FALSE;
    PumpA := FALSE;
    PumpB := FALSE;
END_IF;
IF NOT PowerOK THEN
    MotorRun := FALSE;
    InletValve := FALSE;
END_IF;
END_PROGRAM
"""

BARE = _prog("    x : INT;", "x := x + 1;")

# (description, program, requirement id, expected result)
TRICKY = [
    ("EStop handled only inside a comment",
     _prog("    EStop, Out : BOOL;", "(* IF EStop THEN Out := FALSE; END_IF; *)\nOut := TRUE;"), "emergency_stop", False),
    ("EStop test is inverted (IF NOT EStop)",
     _prog("    EStop, Out : BOOL;", "IF NOT EStop THEN\n    Out := FALSE;\nEND_IF;"), "emergency_stop", False),
    ("EStop never declared",
     _prog("    Out : BOOL;", "IF EStop THEN\n    Out := FALSE;\nEND_IF;"), "emergency_stop", False),
    ("EStop gated inline with AND NOT",
     _prog("    EStop, Cmd, Out : BOOL;", "Out := Cmd AND NOT EStop;"), "emergency_stop", True),
    ("WHILE loop never increments its counter",
     _prog("    n, y : INT;", "WHILE n < 3 DO\n    y := y + 1;\nEND_WHILE;"), "while_loop", False),
    ("State machine with only two states",
     _prog("    State : INT;", "CASE State OF\n    0: State := 1;\n    1: State := 0;\nEND_CASE;"),
     "state_machine", False),
    ("PID output never clamped",
     _prog("    Setpoint, ProcessValue, Kp, Ki, Kd, Error, PrevError, Integral, ControlOutput : REAL;",
           "Error := Setpoint - ProcessValue;\nIntegral := Integral + Ki * Error;\nControlOutput := Kp * Error + Integral;"),
     "pid_loop", False),
    ("VULN marker present in a comment",
     _prog("    x : INT;", "(* VULN: unvalidated setpoint *)\nx := 1;"), "vuln_marker", True),
    ("Power loss handled with NOT PowerOK",
     _prog("    PowerOK, Out : BOOL;", "IF NOT PowerOK THEN\n    Out := FALSE;\nEND_IF;"), "power_loss", True),
    ("Power loss test is not negated",
     _prog("    PowerOK, Out : BOOL;", "IF PowerOK THEN\n    Out := FALSE;\nEND_IF;"), "power_loss", False),
]


IRONPLC_FIXTURES = [
    (r"""error[P0002]: Syntax error
   ┌─ program.st:13:1
   │
13 │ VAR
   │ ^^^ Expected ' ' (space) | '\t' (tab) | '(* ... *)' (comment) | 'CONFIGURATION' | 'FUNCTION' | 'FUNCTION_BLOCK' | 'INTERFACE' | 'PROGRAM' | 'TYPE' | 'VAR_GLOBAL' | '\n' (new line) | '{ ... }' (pragma). Found text 'VAR' that matched token 'VAR'
   │
   = Learn more: https://www.ironplc.com/reference/compiler/problems/P0002.html?version=0.241.0&channel=cli

Error: "Check failed with 1 problem(s)"
""", "P0002 found 'VAR'", "PROGRAM Main"),
    ("""error[P4007]: Variable not defined before used (variable=n)
   ┌─ program.st:40:12
   │
40 │         IF n < 10 THEN
   │            ^ Undefined variable
""", "P4007 variable", "'n : <type>;'"),
    ("""error[P4012]: Function block invocation is not a variable in scope (invocation=Inst)
   ┌─ program.st:30:5
""", "P4012 invocation", "'Inst : <FunctionBlockName>;'"),
]

FB_BODY_INSIDE_PROGRAM = """PROGRAM Main
VAR
    Ctl1 : MyBlock;
END_VAR
VAR_INPUT
    In1 : BOOL;
END_VAR
VAR_OUTPUT
    Out1 : BOOL;
END_VAR
Ctl1(In1 := In1);
END_PROGRAM
"""

FB_NEVER_CALLED = """FUNCTION_BLOCK MyBlock VAR_INPUT In1 : BOOL; END_VAR VAR_OUTPUT Out1 : BOOL; END_VAR Out1 := In1; END_FUNCTION_BLOCK
PROGRAM Main
VAR
    Ctl1 : MyBlock;
    AutoMode : BOOL;
END_VAR
Ctl1.In1 := AutoMode;
END_PROGRAM
"""

ALL_VALUES = (CONTROL_STRATEGIES + IEC_FEATURES + SAFETY_FEATURES + FAILURE_SCENARIOS
              + OPERATING_MODES + SECURITY_PROFILES)


def selftest(compiler_cmd="ironplcc check", use_compiler=True):
    bad = 0
    all_reqs = reqs_for(ALL_VALUES)
    by_id = {r.id: r for r in all_reqs}

    print(f"[A] Checker logic ({len(all_reqs)} requirement checks)")
    passed, failed = run_checks(KITCHEN_SINK, all_reqs)
    print(f"  reference program passes {len(passed)}/{len(all_reqs)}"
          + ("" if not failed else "   FAILED: " + ", ".join(r.id for r in failed)))
    bad += len(failed)

    passed, _ = run_checks(BARE, all_reqs)
    wrongly = [r.id for r in passed if r.id != "structure"]
    print(f"  bare program passes only 'structure': {'yes' if not wrongly else 'NO, also ' + ', '.join(wrongly)}")
    bad += len(wrongly)

    for desc, code, rid, expect in TRICKY:
        got = bool(run_checks(code, [by_id[rid]])[0])
        mark = "ok  " if got == expect else "FAIL"
        bad += got != expect
        print(f"  {mark} {desc}: {rid} -> {'pass' if got else 'fail'} (expected {'pass' if expect else 'fail'})")

    fb_req = by_id["function_block"]
    h = {}
    fb_ok = not run_checks(FB_NEVER_CALLED, [fb_req], h)[0] and "never called" in h.get("function_block", "") \
        and "Ctl1(In1 := <value>)" in h["function_block"]
    bad += not fb_ok
    print(f"  {'ok  ' if fb_ok else 'FAIL'} FB instance set but never called -> {h.get('function_block', '')[:80]}...")
    h2 = {}
    run_checks(FB_BODY_INSIDE_PROGRAM, [fb_req], h2)
    inside_ok = "BEFORE PROGRAM Main" in h2.get("function_block", "")
    bad += not inside_ok
    print(f"  {'ok  ' if inside_ok else 'FAIL'} FB body dumped inside PROGRAM -> {h2.get('function_block', '')[:80]}...")
    print("  compiler-message translation:")
    for out, key, must_contain in IRONPLC_FIXTURES:
        ps = parse_compiler(out)
        got_key, hint = (problem_key(ps[0]), hint_for(ps[0])) if ps else ("", "")
        good = got_key == key and must_contain in hint
        bad += not good
        print(f"  {'ok  ' if good else 'FAIL'} {key} -> {hint[:95]}{'...' if len(hint) > 95 else ''}")

    if use_compiler:
        print(f"\n[B] Compiler gate: {compiler_cmd}")
        try:
            unsupported, gate_ok, details = probe_compiler(compiler_cmd)
        except CompilerMissing as e:
            print("  " + str(e))
            return 1
        for name, (ok, out) in details.items():
            label = name.replace("bad:", "rejects invalid: ")
            print(f"  {'ok  ' if ok else 'FAIL'} {label}")
            if not ok and out:
                print("       " + out.splitlines()[0][:150])
        ks_ok, ks_out = compile_check(KITCHEN_SINK, compiler_cmd)
        print(f"  {'ok  ' if ks_ok else 'info'} reference program compiles")
        if not ks_ok:
            print("       " + "\n       ".join(ks_out.splitlines()[:6]))
        if not gate_ok:
            print("  The compiler gate is not trustworthy (accepts invalid code or rejects the minimal program).")
            bad += 1
        if unsupported - {"minimal"}:
            excluded = sorted(v for v, deps in CAPABILITY_DEPS.items() if deps & unsupported)
            print(f"  Unsupported by this ironplcc: {', '.join(sorted(unsupported))}")
            print(f"  agent.py will not sample: {', '.join(excluded)}")
    else:
        print("\n[B] Compiler gate skipped (--no-compiler)")

    print("\nSELFTEST " + ("PASSED" if bad == 0 else f"FAILED ({bad} problems)"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-compiler", action="store_true", help="selftest checker logic only")
    ap.add_argument("--compiler", default="ironplcc check")
    ap.add_argument("--show-spec", action="store_true", help="print one random generator prompt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.show_spec:
        print(spec_prompt(build_spec(random.Random(a.seed))))
    elif a.selftest:
        sys.exit(selftest(a.compiler, use_compiler=not a.no_compiler))
    else:
        ap.print_help()
