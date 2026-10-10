# System Prompt

You are a Verilog autocomplete engine.
Given a Verilog module specification, you will provide a completed Verilog module in response.
Do not output any reasoning.
Do not include explanatory comments in the module.

# 4-bit Adder

## Design Prompt

Design a Verilog module that takes two 4-bit inputs, a and b, and outputs a 4-bit output sum and a 1-bit carryout, cout.

## Attempts

This module was zero-shotted.

## Module

```verilog
module adder (
    input [3:0] a,
    input [3:0] b,
    output reg [3:0] sum,
    output reg cout
);

    always @(*) begin
        {cout, sum} = a + b;
    end

endmodule
```

## Testbench

```verilog
`timescale 1ns/1ps

// Self-checking exhaustive testbench for a 4-bit adder.
//
// Assumed DUT interface (rename ports in the instantiation if yours differ):
//   module adder (
//     input  [3:0] A, B,
//     output [3:0] sum,
//     output       cout
//   );

module adder_tb;

    reg  [3:0] A, B;
    wire [3:0] sum;
    wire       cout;

    integer errors = 0;
    integer checks = 0;

    adder dut (
        .a    (A),
        .b    (B),
        .sum  (sum),
        .cout (cout)
    );

    reg [4:0] expected;

    task check;
        input [3:0] a;
        input [3:0] b;
        begin
            A = a; B = b;
            #1;
            expected = {1'b0, a} + {1'b0, b};
            checks = checks + 1;
            if ({cout, sum} !== expected) begin
                errors = errors + 1;
                $display("FAIL t=%0t A=%h B=%h | got cout=%b sum=%h | exp cout=%b sum=%h",
                         $time, a, b, cout, sum, expected[4], expected[3:0]);
            end
        end
    endtask

    integer i, j;

    initial begin
        $dumpfile("adder_tb.vcd");
        $dumpvars(0, adder_tb);

        // Directed corner cases
        check(4'h0, 4'h0);   // zero
        check(4'h0, 4'hF);
        check(4'hF, 4'h0);
        check(4'hF, 4'h1);   // carry out, wraps to zero
        check(4'hF, 4'hF);   // max + max
        check(4'h7, 4'h8);   // all ones, no carry
        check(4'h8, 4'h8);   // MSB-only carry
        check(4'h5, 4'hA);   // complementary, no carry
        check(4'h1, 4'h1);

        // Exhaustive: every A x every B (256 checks)
        for (i = 0; i < 16; i = i + 1)
            for (j = 0; j < 16; j = j + 1)
                check(i[3:0], j[3:0]);

        $display("----------------------------------------");
        if (errors == 0)
            $display("PASS: %0d checks, 0 errors", checks);
        else
            $display("FAIL: %0d checks, %0d errors", checks, errors);
        $display("----------------------------------------");
        $finish;
    end

endmodule
```

# 4-bit ALU

## Design Prompt

Design a 4-bit ALU that supports the following operations: ADD, SUB, AND, OR.
It should take two 4-bit inputs A and B, a 2-bit operation code (opcode), and produce a 4-bit output Y.
It should output a carry flag and a zero flag.
The opcode should be defined as follows: 00 for ADD, 01 for SUB, 10 for AND, and 11 for OR.

## Attempts

This module was zero-shotted.

## Module

```verilog
module alu (
    input [3:0] A,
    input [3:0] B,
    input [1:0] opcode,
    output reg [3:0] Y,
    output reg carry,
    output reg zero
);

always @(*) begin
    case (opcode)
        2'b00: begin
            {carry, Y} = A + B;
            zero = (Y == 4'b0000);
        end
        2'b01: begin
            {carry, Y} = A - B;
            zero = (Y == 4'b0000);
        end
        2'b10: begin
            Y = A & B;
            zero = (Y == 4'b0000);
            carry = 1'b0;
        end
        2'b11: begin
            Y = A | B;
            zero = (Y == 4'b0000);
            carry = 1'b0;
        end
    endcase
end

endmodule
```

## Testbench

```verilog
`timescale 1ns/1ps

// Self-checking exhaustive testbench for a 4-bit ALU.
//
// Assumed DUT interface (rename ports below if yours differ):
//   module alu (
//     input  [3:0] A, B,
//     input  [1:0] opcode,   // 00 = ADD, 01 = SUB, 10 = AND, 11 = OR
//     output [3:0] Y,
//     output       carry,
//     output       zero
//   );
//
// Flag conventions checked here:
//   ADD: carry = carry-out of A + B
//   SUB: carry = borrow (1 when A < B), i.e. bit 4 of A - B in 5-bit arithmetic
//        (set SUB_CARRY_IS_NO_BORROW = 1 if your ALU uses A + ~B + 1 carry-out)
//   AND/OR: carry = 0
//   zero = 1 when Y == 0 (all ops)

module alu_tb;

    localparam [1:0] OP_ADD = 2'b00,
                     OP_SUB = 2'b01,
                     OP_AND = 2'b10,
                     OP_OR  = 2'b11;

    localparam SUB_CARRY_IS_NO_BORROW = 0;

    reg  [3:0] A, B;
    reg  [1:0] opcode;
    wire [3:0] Y;
    wire       carry, zero;

    integer errors = 0;
    integer checks = 0;

    alu dut (
        .A      (A),
        .B      (B),
        .opcode (opcode),
        .Y      (Y),
        .carry  (carry),
        .zero   (zero)
    );

    // Reference model
    reg [4:0] exp_full;
    reg [3:0] exp_y;
    reg       exp_carry;
    reg       exp_zero;

    task compute_expected;
        input [3:0] a;
        input [3:0] b;
        input [1:0] op;
        begin
            exp_carry = 1'b0;
            case (op)
                OP_ADD: begin
                    exp_full  = {1'b0, a} + {1'b0, b};
                    exp_y     = exp_full[3:0];
                    exp_carry = exp_full[4];
                end
                OP_SUB: begin
                    exp_full  = {1'b0, a} - {1'b0, b};
                    exp_y     = exp_full[3:0];
                    exp_carry = SUB_CARRY_IS_NO_BORROW ? ~exp_full[4] : exp_full[4];
                end
                OP_AND: exp_y = a & b;
                OP_OR : exp_y = a | b;
                default: exp_y = 4'bx;
            endcase
            exp_zero = (exp_y == 4'b0000);
        end
    endtask

    task check;
        input [3:0] a;
        input [3:0] b;
        input [1:0] op;
        begin
            A = a; B = b; opcode = op;
            #1;
            compute_expected(a, b, op);
            checks = checks + 1;
            if (Y !== exp_y || carry !== exp_carry || zero !== exp_zero) begin
                errors = errors + 1;
                $display("FAIL t=%0t op=%b A=%h B=%h | got Y=%h C=%b Z=%b | exp Y=%h C=%b Z=%b",
                         $time, op, a, b, Y, carry, zero, exp_y, exp_carry, exp_zero);
            end
        end
    endtask

    integer i, j, k;

    initial begin
        $dumpfile("alu_tb.vcd");
        $dumpvars(0, alu_tb);

        // Directed corner cases
        check(4'h0, 4'h0, OP_ADD);   // zero result
        check(4'hF, 4'h1, OP_ADD);   // carry out, wraps to zero
        check(4'hF, 4'hF, OP_ADD);   // max + max
        check(4'h7, 4'h8, OP_ADD);   // no carry, all ones
        check(4'h5, 4'h5, OP_SUB);   // equal -> zero, no borrow
        check(4'h0, 4'h1, OP_SUB);   // borrow, wraps to F
        check(4'h3, 4'h7, OP_SUB);   // negative result
        check(4'hA, 4'h5, OP_AND);   // disjoint bits -> zero
        check(4'hF, 4'hF, OP_AND);
        check(4'h0, 4'h0, OP_OR);    // zero result
        check(4'hA, 4'h5, OP_OR);    // complementary -> F

        // Exhaustive: every opcode x every A x every B
        for (k = 0; k < 4; k = k + 1)
            for (i = 0; i < 16; i = i + 1)
                for (j = 0; j < 16; j = j + 1)
                    check(i[3:0], j[3:0], k[1:0]);

        $display("----------------------------------------");
        if (errors == 0)
            $display("PASS: %0d checks, 0 errors", checks);
        else
            $display("FAIL: %0d checks, %0d errors", checks, errors);
        $display("----------------------------------------");
        $finish;
    end

endmodule
```

# 8-bit ALU (v1)

This iteration of the 8-bit ALU included an arithmetic right shift opcode, which created compilation errors.

## Design Prompt

Design an 8-bit ALU that supports the following operations: ADD, SUB, AND, OR, XOR, SLL (logical left shift), SRL (logical right shift), and SRA (arithmetic right shift).
It should take two 8-bit inputs A and B, a 3-bit operation code (opcode), and produce an 8-bit output Y.
It should output a carry flag and a zero flag.
The opcode should be defined as follows: 000 for ADD, 001 for SUB, 010 for AND, 011 for OR, 100 for XOR, 101 for SLL, 110 for SRL, and 111 for SRA.

## Attempts

This module never successfully compiled. Two trials of four iterations each all produced the same error, that being a compile error on the arithmetic right shift at line 41.

## Module

```verilog
module alu (
    input [7:0] A,
    input [7:0] B,
    input [2:0] opcode,
    output reg [7:0] Y,
    output reg carry,
    output reg zero
);

always @(*) begin
    case (opcode)
        3'b000: begin // ADD
            {carry, Y} = A + B;
            zero = (Y == 8'b0);
        end
        3'b001: begin // SUB
            {carry, Y} = A - B;
            zero = (Y == 8'b0);
        end
        3'b010: begin // AND
            Y = A & B;
            zero = (Y == 8'b0);
        end
        3'b011: begin // OR
            Y = A | B;
            zero = (Y == 8'b0);
        end
        3'b100: begin // XOR
            Y = A ^ B;
            zero = (Y == 8'b0);
        end
        3'b101: begin // SLL
            Y = A << B[2:0];
            zero = (Y == 8'b0);
        end
        3'b110: begin // SRL
            Y = A >> B[2:0];
            zero = (Y == 8'b0);
        end
        3'b111: begin // SRA
            Y = (A[7] ? ({8{A[7]}}) | (A >> B[2:0])) : (A >> B[2:0]);
            zero = (Y == 8'b0);
        end
    endcase
end

endmodule
```

## Testbench

```verilog
`timescale 1ns/1ps

// Self-checking exhaustive testbench for an 8-bit ALU.
//
// Assumed DUT interface (rename ports in the instantiation if yours differ):
//   module alu8 (
//     input  [7:0] A, B,
//     input  [2:0] opcode,   // 000 ADD, 001 SUB, 010 AND, 011 OR,
//                            // 100 XOR, 101 SLL, 110 SRL, 111 SRA
//     output [7:0] Y,
//     output       carry,
//     output       zero
//   );
//
// Conventions checked (adjust the parameters below to match your design):
//   ADD       : carry = carry-out of A + B
//   SUB       : carry = borrow (1 when A < B); set SUB_CARRY_IS_NO_BORROW = 1
//               if your ALU uses A + ~B + 1 and reports its carry-out instead
//   AND/OR/XOR: carry = 0
//   Shifts    : shift amount = B[2:0]; Y shifts A
//               SHIFT_CARRY_MODE = 0 -> carry must be 0
//                                  1 -> carry = last bit shifted out (0 if amount is 0)
//                                  2 -> carry is not checked for shifts
//   zero      : 1 when Y == 0, for every operation

module alu_tb;

    localparam [2:0] OP_ADD = 3'b000,
                     OP_SUB = 3'b001,
                     OP_AND = 3'b010,
                     OP_OR  = 3'b011,
                     OP_XOR = 3'b100,
                     OP_SLL = 3'b101,
                     OP_SRL = 3'b110,
                     OP_SRA = 3'b111;

    localparam SUB_CARRY_IS_NO_BORROW = 0;
    localparam SHIFT_CARRY_MODE       = 0;

    reg  [7:0] A, B;
    reg  [2:0] opcode;
    wire [7:0] Y;
    wire       carry, zero;

    integer errors = 0;
    integer checks = 0;

    alu8 dut (
        .A      (A),
        .B      (B),
        .opcode (opcode),
        .Y      (Y),
        .carry  (carry),
        .zero   (zero)
    );

    // Reference model
    reg [8:0] exp_full;
    reg [7:0] exp_y;
    reg       exp_carry;
    reg       exp_zero;
    reg       check_carry;
    reg [2:0] shamt;
    reg signed [7:0] a_signed;

    task compute_expected;
        input [7:0] a;
        input [7:0] b;
        input [2:0] op;
        begin
            exp_carry   = 1'b0;
            check_carry = 1'b1;
            shamt       = b[2:0];
            a_signed    = a;
            case (op)
                OP_ADD: begin
                    exp_full  = {1'b0, a} + {1'b0, b};
                    exp_y     = exp_full[7:0];
                    exp_carry = exp_full[8];
                end
                OP_SUB: begin
                    exp_full  = {1'b0, a} - {1'b0, b};
                    exp_y     = exp_full[7:0];
                    exp_carry = SUB_CARRY_IS_NO_BORROW ? ~exp_full[8] : exp_full[8];
                end
                OP_AND: exp_y = a & b;
                OP_OR : exp_y = a | b;
                OP_XOR: exp_y = a ^ b;
                OP_SLL: begin
                    exp_y = a << shamt;
                    if (SHIFT_CARRY_MODE == 1)
                        exp_carry = (shamt == 0) ? 1'b0 : a[8 - shamt];
                    if (SHIFT_CARRY_MODE == 2) check_carry = 1'b0;
                end
                OP_SRL: begin
                    exp_y = a >> shamt;
                    if (SHIFT_CARRY_MODE == 1)
                        exp_carry = (shamt == 0) ? 1'b0 : a[shamt - 1];
                    if (SHIFT_CARRY_MODE == 2) check_carry = 1'b0;
                end
                OP_SRA: begin
                    exp_y = a_signed >>> shamt;
                    if (SHIFT_CARRY_MODE == 1)
                        exp_carry = (shamt == 0) ? 1'b0 : a[shamt - 1];
                    if (SHIFT_CARRY_MODE == 2) check_carry = 1'b0;
                end
                default: exp_y = 8'bx;
            endcase
            exp_zero = (exp_y == 8'h00);
        end
    endtask

    task check;
        input [7:0] a;
        input [7:0] b;
        input [2:0] op;
        begin
            A = a; B = b; opcode = op;
            #1;
            compute_expected(a, b, op);
            checks = checks + 1;
            if (Y !== exp_y || zero !== exp_zero ||
                (check_carry && carry !== exp_carry)) begin
                errors = errors + 1;
                if (errors <= 50)
                    $display("FAIL t=%0t op=%b A=%h B=%h | got Y=%h C=%b Z=%b | exp Y=%h C=%b Z=%b",
                             $time, op, a, b, Y, carry, zero, exp_y, exp_carry, exp_zero);
            end
        end
    endtask

    integer i, j, k;

    initial begin
        $dumpfile("alu8_tb.vcd");
        $dumpvars(0, alu8_tb);

        // Directed corner cases
        check(8'h00, 8'h00, OP_ADD);   // zero result
        check(8'hFF, 8'h01, OP_ADD);   // carry out, wraps to zero
        check(8'hFF, 8'hFF, OP_ADD);   // max + max
        check(8'h7F, 8'h01, OP_ADD);   // signed overflow boundary
        check(8'h55, 8'h55, OP_SUB);   // equal -> zero, no borrow
        check(8'h00, 8'h01, OP_SUB);   // borrow, wraps to FF
        check(8'h80, 8'h01, OP_SUB);
        check(8'hF0, 8'h0F, OP_AND);   // disjoint -> zero
        check(8'hAA, 8'h55, OP_OR);    // complementary -> FF
        check(8'hFF, 8'hFF, OP_XOR);   // identical -> zero
        check(8'h81, 8'h01, OP_SLL);   // MSB shifted out
        check(8'h81, 8'h07, OP_SLL);   // max shift amount
        check(8'h81, 8'h00, OP_SLL);   // zero shift amount
        check(8'h81, 8'h01, OP_SRL);   // LSB shifted out, zero fill
        check(8'h80, 8'h07, OP_SRL);
        check(8'h80, 8'h01, OP_SRA);   // sign extension
        check(8'h80, 8'h07, OP_SRA);   // -> FF
        check(8'h7F, 8'h07, OP_SRA);   // positive, zero fill
        check(8'hF0, 8'hF3, OP_SRA);   // upper bits of B ignored

        // Exhaustive: every opcode x every A x every B (524,288 checks)
        for (k = 0; k < 8; k = k + 1)
            for (i = 0; i < 256; i = i + 1)
                for (j = 0; j < 256; j = j + 1)
                    check(i[7:0], j[7:0], k[2:0]);

        $display("----------------------------------------");
        if (errors == 0)
            $display("PASS: %0d checks, 0 errors", checks);
        else
            $display("FAIL: %0d checks, %0d errors (first 50 shown)", checks, errors);
        $display("----------------------------------------");
        $finish;
    end

endmodule
```

# 8-bit ALU (v2)

This iteration of the ALU replaced the SRA op with NOT_A.

## Design Prompt

Design an 8-bit ALU that supports the following operations: ADD, SUB, AND, OR, XOR, SLL (logical left shift), SRL (logical right shift), and NOT_A.
It should take two 8-bit inputs A and B, a 3-bit operation code (opcode), and produce an 8-bit output Y.
It should output a carry flag "carry" and a zero flag "zero".
SHL and SHR should shift A based on the lower bits of B.
The opcode should be defined as follows: 000 for ADD, 001 for SUB, 010 for AND, 011 for OR, 100 for XOR, 101 for SLL, 110 for SRL, and 111 for NOT_A.

## Attempts

On the first few attempts of this module, there were discrepancies between the behavior of the shifts on the module and testbench. This resulted in several thousand errors in the test suite.

After modifying the prompt to be consistent with the testbench's behavior, a correct module was successfully produced on the first iteration.

## Module
```verilog
module alu (
    input [7:0] A,
    input [7:0] B,
    input [2:0] opcode,
    output reg [7:0] Y,
    output reg carry,
    output reg zero
);

always @(*) begin
    case (opcode)
        3'b000: begin // ADD
            {carry, Y} = A + B;
            zero = (Y == 8'b0);
        end
        3'b001: begin // SUB
            {carry, Y} = A - B;
            zero = (Y == 8'b0);
        end
        3'b010: begin // AND
            Y = A & B;
            zero = (Y == 8'b0);
        end
        3'b011: begin // OR
            Y = A | B;
            zero = (Y == 8'b0);
        end
        3'b100: begin // XOR
            Y = A ^ B;
            zero = (Y == 8'b0);
        end
        3'b101: begin // SLL
            Y = A << B[2:0];
            zero = (Y == 8'b0);
        end
        3'b110: begin // SRL
            Y = A >> B[2:0];
            zero = (Y == 8'b0);
        end
        3'b111: begin // NOT_A
            Y = ~A;
            zero = (Y == 8'b0);
        end
    endcase
end

endmodule
```

## Testbench

```verilog
`timescale 1ns/1ps

// Self-checking exhaustive testbench for an 8-bit ALU.
//
// Assumed DUT interface (rename ports in the instantiation if yours differ):
//   module alu (
//     input  [7:0] A, B,
//     input  [2:0] opcode,   // 000 ADD, 001 SUB, 010 AND, 011 OR,
//                            // 100 XOR, 101 SLL, 110 SRL, 111 NOT
//     output [7:0] Y,
//     output       carry,
//     output       zero
//   );
//
// Conventions checked (adjust the parameters below to match your design):
//   ADD       : carry = carry-out of A + B
//   SUB       : carry = borrow (1 when A < B); set SUB_CARRY_IS_NO_BORROW = 1
//               if your ALU uses A + ~B + 1 and reports its carry-out instead
//   AND/OR/XOR: carry = 0
//   NOT       : Y = ~A (B is ignored), carry = 0
//   Shifts    : shift amount = B[2:0]; Y shifts A
//               SHIFT_CARRY_MODE = 0 -> carry must be 0
//                                  1 -> carry = last bit shifted out (0 if amount is 0)
//                                  2 -> carry is not checked for shifts
//   zero      : 1 when Y == 0, for every operation

module alu_tb;

    localparam [2:0] OP_ADD = 3'b000,
                     OP_SUB = 3'b001,
                     OP_AND = 3'b010,
                     OP_OR  = 3'b011,
                     OP_XOR = 3'b100,
                     OP_SLL = 3'b101,
                     OP_SRL = 3'b110,
                     OP_NOT = 3'b111;

    localparam SUB_CARRY_IS_NO_BORROW = 0;
    localparam SHIFT_CARRY_MODE       = 0;

    reg  [7:0] A, B;
    reg  [2:0] opcode;
    wire [7:0] Y;
    wire       carry, zero;

    integer errors = 0;
    integer checks = 0;

    alu dut (
        .A      (A),
        .B      (B),
        .opcode (opcode),
        .Y      (Y),
        .carry  (carry),
        .zero   (zero)
    );

    // Reference model
    reg [8:0] exp_full;
    reg [7:0] exp_y;
    reg       exp_carry;
    reg       exp_zero;
    reg       check_carry;
    reg [2:0] shamt;

    task compute_expected;
        input [7:0] a;
        input [7:0] b;
        input [2:0] op;
        begin
            exp_carry   = 1'b0;
            check_carry = 1'b1;
            shamt       = b[2:0];
            case (op)
                OP_ADD: begin
                    exp_full  = {1'b0, a} + {1'b0, b};
                    exp_y     = exp_full[7:0];
                    exp_carry = exp_full[8];
                end
                OP_SUB: begin
                    exp_full  = {1'b0, a} - {1'b0, b};
                    exp_y     = exp_full[7:0];
                    exp_carry = SUB_CARRY_IS_NO_BORROW ? ~exp_full[8] : exp_full[8];
                end
                OP_AND: exp_y = a & b;
                OP_OR : exp_y = a | b;
                OP_XOR: exp_y = a ^ b;
                OP_SLL: begin
                    exp_y = a << shamt;
                    if (SHIFT_CARRY_MODE == 1)
                        exp_carry = (shamt == 0) ? 1'b0 : a[8 - shamt];
                    if (SHIFT_CARRY_MODE == 2) check_carry = 1'b0;
                end
                OP_SRL: begin
                    exp_y = a >> shamt;
                    if (SHIFT_CARRY_MODE == 1)
                        exp_carry = (shamt == 0) ? 1'b0 : a[shamt - 1];
                    if (SHIFT_CARRY_MODE == 2) check_carry = 1'b0;
                end
                OP_NOT: exp_y = ~a;
                default: exp_y = 8'bx;
            endcase
            exp_zero = (exp_y == 8'h00);
        end
    endtask

    task check;
        input [7:0] a;
        input [7:0] b;
        input [2:0] op;
        begin
            A = a; B = b; opcode = op;
            #1;
            compute_expected(a, b, op);
            checks = checks + 1;
            if (Y !== exp_y || zero !== exp_zero ||
                (check_carry && carry !== exp_carry)) begin
                errors = errors + 1;
                if (errors <= 50)
                    $display("FAIL t=%0t op=%b A=%h B=%h | got Y=%h C=%b Z=%b | exp Y=%h C=%b Z=%b",
                             $time, op, a, b, Y, carry, zero, exp_y, exp_carry, exp_zero);
            end
        end
    endtask

    integer i, j, k;

    initial begin
        $dumpfile("alu_tb.vcd");
        $dumpvars(0, alu_tb);

        // Directed corner cases
        check(8'h00, 8'h00, OP_ADD);   // zero result
        check(8'hFF, 8'h01, OP_ADD);   // carry out, wraps to zero
        check(8'hFF, 8'hFF, OP_ADD);   // max + max
        check(8'h7F, 8'h01, OP_ADD);   // signed overflow boundary
        check(8'h55, 8'h55, OP_SUB);   // equal -> zero, no borrow
        check(8'h00, 8'h01, OP_SUB);   // borrow, wraps to FF
        check(8'h80, 8'h01, OP_SUB);
        check(8'hF0, 8'h0F, OP_AND);   // disjoint -> zero
        check(8'hAA, 8'h55, OP_OR);    // complementary -> FF
        check(8'hFF, 8'hFF, OP_XOR);   // identical -> zero
        check(8'h81, 8'h01, OP_SLL);   // MSB shifted out
        check(8'h81, 8'h07, OP_SLL);   // max shift amount
        check(8'h81, 8'h00, OP_SLL);   // zero shift amount
        check(8'h81, 8'h01, OP_SRL);   // LSB shifted out, zero fill
        check(8'h80, 8'h07, OP_SRL);
        check(8'h00, 8'h00, OP_NOT);   // -> FF
        check(8'hFF, 8'h00, OP_NOT);   // -> zero result
        check(8'hAA, 8'h00, OP_NOT);   // -> 55
        check(8'h55, 8'hFF, OP_NOT);   // B must be ignored

        // Exhaustive: every opcode x every A x every B (524,288 checks)
        for (k = 0; k < 8; k = k + 1)
            for (i = 0; i < 256; i = i + 1)
                for (j = 0; j < 256; j = j + 1)
                    check(i[7:0], j[7:0], k[2:0]);

        if (errors == 0)
            $display("PASS: %0d checks, 0 errors", checks);
        else
            $display("FAIL: %0d checks, %0d errors (first 50 shown)", checks, errors);
        $finish;
    end

endmodule
```
