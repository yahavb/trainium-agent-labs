#!/usr/bin/env python3
"""Hard test pack for the testbench: 9 circuits, each with a CORRECT and a BUGGY version.

  python make_tests.py          -> writes tests/<name>.v and tests/<name>_bug.v, prints the commands
  python make_tests.py --run    -> runs your testbench on all of them and prints a score table

A perfect testbench: PASSES every correct design and CATCHES every buggy one.
"""
import os, sys, json, shutil

T = {}

T["mult8"] = dict(level="2 big", spec=
"8-bit by 8-bit unsigned multiplier. module mult8(input [7:0] a, input [7:0] b, output [15:0] p); p = a * b.",
ok="""module mult8(input [7:0] a, input [7:0] b, output [15:0] p);
  assign p = a * b;
endmodule
""", bug="""module mult8(input [7:0] a, input [7:0] b, output [15:0] p);
  assign p = a * b[6:0];   // BUG: ignores the top bit of b
endmodule
""")

T["barrel8"] = dict(level="2 big", spec=
"8-bit logical barrel shifter, zeros shifted in. module barrel8(input [7:0] data, input [2:0] shamt, input dir, output [7:0] y); dir=0: y = data shifted LEFT by shamt; dir=1: y = data shifted RIGHT by shamt.",
ok="""module barrel8(input [7:0] data, input [2:0] shamt, input dir, output [7:0] y);
  assign y = dir ? (data >> shamt) : (data << shamt);
endmodule
""", bug="""module barrel8(input [7:0] data, input [2:0] shamt, input dir, output [7:0] y);
  assign y = dir ? (data << shamt) : (data >> shamt);   // BUG: directions swapped
endmodule
""")

T["add16"] = dict(level="2 big", spec=
"16-bit adder with carry. module add16(input [15:0] a, input [15:0] b, input cin, output [15:0] sum, output cout); {cout, sum} = a + b + cin.",
ok="""module add16(input [15:0] a, input [15:0] b, input cin, output [15:0] sum, output cout);
  assign {cout, sum} = a + b + cin;
endmodule
""", bug="""module add16(input [15:0] a, input [15:0] b, input cin, output [15:0] sum, output cout);
  assign sum = a + b + cin;
  assign cout = 1'b0;      // BUG: carry out is lost
endmodule
""")

T["prienc8"] = dict(level="3 tricky", spec=
"8-bit priority encoder. module prienc8(input [7:0] req, output reg [2:0] idx, output valid); idx = position of the HIGHEST set bit of req; valid = 1 if any bit of req is 1; when req is 0, idx = 0 and valid = 0.",
ok="""module prienc8(input [7:0] req, output reg [2:0] idx, output valid);
  integer i;
  assign valid = |req;
  always @(*) begin
    idx = 3'd0;
    for (i = 0; i < 8; i = i + 1)
      if (req[i]) idx = i;
  end
endmodule
""", bug="""module prienc8(input [7:0] req, output reg [2:0] idx, output valid);
  integer i;
  assign valid = |req;
  always @(*) begin
    idx = 3'd0;
    for (i = 7; i >= 0; i = i - 1)   // BUG: lowest set bit wins instead of highest
      if (req[i]) idx = i;
  end
endmodule
""")

T["popcount8"] = dict(level="3 tricky", spec=
"8-bit population count. module popcount8(input [7:0] x, output [3:0] count); count = number of 1 bits in x.",
ok="""module popcount8(input [7:0] x, output [3:0] count);
  assign count = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];
endmodule
""", bug="""module popcount8(input [7:0] x, output [3:0] count);
  assign count = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6];   // BUG: forgets bit 7
endmodule
""")

T["bcd7seg"] = dict(level="3 tricky", spec=
"BCD to 7-segment decoder, active-high segments. module bcd7seg(input [3:0] bcd, output reg [6:0] seg); seg = {g,f,e,d,c,b,a}. Standard digits 0-9 where 6 includes segment a, 7 lights only a,b,c, and 9 includes segment d. Inputs 10-15 give seg = 0.",
ok="""module bcd7seg(input [3:0] bcd, output reg [6:0] seg);
  always @(*) begin
    case (bcd)
      4'd0: seg = 7'h3F;  4'd1: seg = 7'h06;  4'd2: seg = 7'h5B;  4'd3: seg = 7'h4F;
      4'd4: seg = 7'h66;  4'd5: seg = 7'h6D;  4'd6: seg = 7'h7D;  4'd7: seg = 7'h07;
      4'd8: seg = 7'h7F;  4'd9: seg = 7'h6F;
      default: seg = 7'h00;
    endcase
  end
endmodule
""", bug="""module bcd7seg(input [3:0] bcd, output reg [6:0] seg);
  always @(*) begin
    case (bcd)
      4'd0: seg = 7'h3F;  4'd1: seg = 7'h06;  4'd2: seg = 7'h5F;  4'd3: seg = 7'h4F;   // BUG: digit 2 wrong
      4'd4: seg = 7'h66;  4'd5: seg = 7'h6D;  4'd6: seg = 7'h7D;  4'd7: seg = 7'h07;
      4'd8: seg = 7'h7F;  4'd9: seg = 7'h6F;
      default: seg = 7'h00;
    endcase
  end
endmodule
""")

T["scmp4"] = dict(level="3 tricky", spec=
"4-bit SIGNED (two's complement) comparator. module scmp4(input signed [3:0] a, input signed [3:0] b, output lt, output eq, output gt); lt = (a < b), eq = (a == b), gt = (a > b), comparing as signed numbers -8..7.",
ok="""module scmp4(input signed [3:0] a, input signed [3:0] b, output lt, output eq, output gt);
  assign lt = a < b;
  assign eq = a == b;
  assign gt = a > b;
endmodule
""", bug="""module scmp4(input [3:0] a, input [3:0] b, output lt, output eq, output gt);   // BUG: unsigned compare
  assign lt = a < b;
  assign eq = a == b;
  assign gt = a > b;
endmodule
""")

T["shreg8"] = dict(level="4 clocked", spec=
"8-bit shift register. module shreg8(input clk, input rst, input sin, output reg [7:0] q); on each rising clk edge: if rst (synchronous, active high) q = 0, else q shifts LEFT by one and sin enters bit 0 (q <= {q[6:0], sin}).",
ok="""module shreg8(input clk, input rst, input sin, output reg [7:0] q);
  always @(posedge clk) begin
    if (rst) q <= 8'd0;
    else     q <= {q[6:0], sin};
  end
endmodule
""", bug="""module shreg8(input clk, input rst, input sin, output reg [7:0] q);
  always @(posedge clk) begin
    if (rst) q <= 8'd0;
    else     q <= {sin, q[7:1]};   // BUG: shifts right
  end
endmodule
""")

T["det101"] = dict(level="4 clocked", spec=
"Overlapping 101 sequence detector. module det101(input clk, input rst, input x, output reg y); x is sampled on each rising clk edge; rst is synchronous active high and clears everything (y = 0, history forgotten). y is a registered output: on the edge where the last three samples of x are 1,0,1 (oldest first), y becomes 1, otherwise y becomes 0. Overlapping: input 1,0,1,0,1 detects twice.",
ok="""module det101(input clk, input rst, input x, output reg y);
  reg [1:0] s;   // the previous two samples of x, older one in s[1]
  always @(posedge clk) begin
    if (rst) begin s <= 2'b00; y <= 1'b0; end
    else begin
      y <= (s == 2'b10) && x;
      s <= {s[0], x};
    end
  end
endmodule
""", bug="""module det101(input clk, input rst, input x, output reg y);
  reg [1:0] s;
  always @(posedge clk) begin
    if (rst) begin s <= 2'b00; y <= 1'b0; end
    else begin
      y <= (s == 2'b10) && !x;   // BUG: detects 100 instead of 101
      s <= {s[0], x};
    end
  end
endmodule
""")

def write():
    os.makedirs("tests", exist_ok=True)
    for n, t in T.items():
        open(f"tests/{n}.v", "w").write(t["ok"])
        open(f"tests/{n}_bug.v", "w").write(t["bug"])
    json.dump({n: t["spec"] for n, t in T.items()}, open("tests/specs.json", "w"), indent=2)
    print(f"wrote {2 * len(T)} files to tests/\n")
    for n, t in T.items():
        print(f'# level {t["level"]}: {n}')
        print(f'python verify.py --design tests/{n}.v --regen --spec "{t["spec"]}"')
        print(f'python verify.py --design tests/{n}_bug.v --spec "{t["spec"]}"\n')

def run(names):
    import verify
    rows = []
    for n in names:
        t = T[n]
        d = os.path.join("designs", n)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        open(os.path.join(d, "spec.txt"), "w").write(t["spec"] + "\n")
        open(os.path.join(d, "design.v"), "w").write(t["ok"])
        good = verify.verify(d, regen=True, print_tb=False)
        bd = d + "_bug"                          # the buggy copy is judged by the SAME testbench
        shutil.rmtree(bd, ignore_errors=True)
        os.makedirs(bd)
        open(os.path.join(bd, "spec.txt"), "w").write(t["spec"] + "\n")
        open(os.path.join(bd, "design.v"), "w").write(t["bug"])
        if os.path.exists(os.path.join(d, "tb.v")):
            shutil.copy(os.path.join(d, "tb.v"), os.path.join(bd, "tb.v"))
            bad = verify.verify(bd, print_tb=False, do_mutation=False)
        else:
            bad = None
        rows.append((n, t["level"], good, bad))

    print("\n================ TESTBENCH SCORECARD ================")
    print(f"{'circuit':10s} {'level':10s} {'testbench':10s} {'correct':8s} {'buggy':12s} mutants")
    perfect = 0
    for n, lvl, g, b in rows:
        tbok = g and g["behaviour_tested"]
        ok = "PASS ✓" if g and g["passed"] else ("REVIEW ⚠" if g and g.get("testbench_suspect") else "FAIL ✗")
        caught = ("caught ✓" if b and not b["passed"] else "MISSED ✗") if tbok else "-"
        mut = f"{g['mutation']['killed']}/{g['mutation']['total']}" if g and g.get("mutation") else "-"
        perfect += bool(tbok and g["passed"] and b and not b["passed"])
        print(f"{n:10s} {lvl:10s} {'built' if tbok else 'NONE':10s} {ok:8s} {caught:12s} {mut}")
    review = sum(1 for _, _, g, _ in rows if g and g.get("testbench_suspect"))
    print(f"\nperfect (testbench built, correct passed, bug caught): {perfect}/{len(rows)}")
    if review:
        print(f"sent to human review instead of wrongly blaming a correct design: {review}")

if __name__ == "__main__":
    if "--run" in sys.argv:
        only = [a for a in sys.argv[1:] if not a.startswith("--")]
        run(only or list(T))
    else:
        write()
