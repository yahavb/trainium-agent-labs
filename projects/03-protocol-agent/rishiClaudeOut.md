# rishiClaudeOut

This file collects the Qwen3-8B Verilog runs for the protocol agent (`projects/03-protocol-agent`). It has four sections: the prompts sent to the model, the Verilog the model wrote (`uart_tx/realworld/`), the golden testbenches, and the checker's verdict on each generated design.

## Contents

1. [Prompts](#prompts)
2. [Generated Verilog files](#generated-verilog-files)
3. [Test benches](#test-benches)
4. [Test bench errors](#test-bench-errors)

## Prompts

### uart_tx (`uart_tx/prompt.txt`)

```text
Write a Verilog-2001 module that transmits one byte over UART (8N1).

Use exactly this header:
module uart_tx #(parameter CLKS_PER_BIT = 16) (
  input  wire       clk,
  input  wire       rst,      // active-high, synchronous
  input  wire [7:0] data_in,
  input  wire       start,    // 1-cycle pulse: send data_in
  output reg        tx,
  output reg        busy,
  output reg        done
);

Behaviour:
- Idle and after reset: tx = 1, busy = 0, done = 0.
- When start is 1 and not busy: copy data_in into a register, set busy = 1.
- Frame on tx: start bit (0), then data bits 0..7 (lowest bit first), then stop bit (1).
- Every bit lasts exactly CLKS_PER_BIT clock cycles. Use the parameter, not a number.
- When the stop bit finishes: busy = 0, and done = 1 for exactly one clock.
- Everything happens in one always @(posedge clk) block with non-blocking assignments (<=).

Reply with only one ```verilog code block containing the complete module.
```

### uart_rx (`uart_rx/prompt.txt`)

```text
Write a Verilog-2001 module that receives one byte at a time over UART (8N1).

Use exactly this header:
module uart_rx #(parameter CLKS_PER_BIT = 16) (
  input  wire       clk,
  input  wire       rst,          // active-high, synchronous
  input  wire       rx,           // serial line, idles at 1
  output reg  [7:0] data_out,
  output reg        data_valid,   // 1 for exactly one clock per good byte
  output reg        frame_error   // 1 for exactly one clock when the stop bit is 0
);

Behaviour:
- After reset: data_out = 0, data_valid = 0, frame_error = 0. Stay idle while rx = 1.
- A frame on rx is: start bit (0), data bits 0..7 (lowest bit first), stop bit (1).
  Every bit lasts CLKS_PER_BIT clock cycles. Use the parameter, not a number.
- When idle and rx is 0: wait CLKS_PER_BIT/2 clocks to reach the middle of the start bit.
  If rx is 1 there, it was noise: go back to idle.
- Otherwise sample rx every CLKS_PER_BIT clocks after that (the middle of each bit):
  8 data bits, lowest bit first, then the stop bit.
- Stop bit is 1: data_out = the received byte and data_valid = 1 for exactly one clock.
- Stop bit is 0: frame_error = 1 for exactly one clock, data_valid stays 0, then wait
  until rx is back to 1 before looking for the next start bit.
- Bytes can arrive back to back (the next start bit begins right when the stop bit ends),
  so go back to idle right after sampling the stop bit.
- rx is already synchronous to clk; no synchronizer is needed.
- Everything happens in one always @(posedge clk) block with non-blocking assignments (<=).

Reply with only one ```verilog code block containing the complete module.
```

## Generated Verilog files

These are Qwen3-8B's actual `uart_tx` designs, and the same designs after each fix, saved in `uart_tx/realworld/`. No `uart_rx` output has been saved yet.

### `qwen3_reload_zero.v`

```verilog
module uart_tx #(
  parameter CLKS_PER_BIT = 16
) (
  input  wire       clk,
  input  wire       rst,      // active-high, synchronous
  input  wire [7:0] data_in,
  input  wire       start,    // 1-cycle pulse: send data_in
  output reg        tx,
  output reg        busy,
  output reg        done
);

  reg [7:0] data_reg;
  reg [4:0] bit_cnt;
  reg [CLKS_PER_BIT-1:0] bit_timer;

  always @(posedge clk) begin
    if (rst) begin
      data_reg <= 8'd0;
      bit_cnt <= 5'd0;
      bit_timer <= {CLKS_PER_BIT{1'b0}};
      tx <= 1'b1;
      busy <= 1'b0;
      done <= 1'b0;
    end else if (start && !busy) begin
      data_reg <= data_in;
      bit_cnt <= 5'd0;
      bit_timer <= {CLKS_PER_BIT{1'b0}};
      tx <= 1'b0;
      busy <= 1'b1;
      done <= 1'b0;
    end else if (busy) begin
      if (bit_timer == 0) begin
        if (bit_cnt == 5'd0) begin
          // Start bit
          tx <= 1'b0;
        end else if (bit_cnt < 5'd8) begin
          // Data bits (LSB first)
          tx <= data_reg[bit_cnt - 1];
        end else begin
          // Stop bit
          tx <= 1'b1;
        end
        bit_cnt <= bit_cnt + 1;
        bit_timer <= {CLKS_PER_BIT{1'b0}};
      end else begin
        bit_timer <= bit_timer - 1;
      end

      if (bit_cnt == 5'd9) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
    end
  end

endmodule
```

### `qwen3_reload_zero_v2.v`

```verilog
module uart_tx #(
  parameter CLKS_PER_BIT = 16
) (
  input  wire       clk,
  input  wire       rst,      // active-high, synchronous
  input  wire [7:0] data_in,
  input  wire       start,    // 1-cycle pulse: send data_in
  output reg        tx,
  output reg        busy,
  output reg        done
);

  reg [7:0] data_reg;
  reg [4:0] bit_cnt;
  reg [CLKS_PER_BIT-1:0] bit_timer;

  always @(posedge clk) begin
    if (rst) begin
      data_reg <= 8'h00;
      bit_cnt <= 5'd0;
      bit_timer <= {CLKS_PER_BIT{1'b0}};
      tx <= 1'b1;
      busy <= 1'b0;
      done <= 1'b0;
    end else if (start && !busy) begin
      data_reg <= data_in;
      bit_cnt <= 5'd0;
      bit_timer <= {CLKS_PER_BIT{1'b0}};
      tx <= 1'b0;
      busy <= 1'b1;
      done <= 1'b0;
    end else if (busy) begin
      if (bit_timer == 0) begin
        if (bit_cnt == 5'd0) begin
          // Start bit
          tx <= 1'b0;
        end else if (bit_cnt < 5'd8) begin
          // Data bits (LSB first)
          tx <= data_reg[bit_cnt - 1];
        end else begin
          // Stop bit
          tx <= 1'b1;
        end
        bit_cnt <= bit_cnt + 1;
        bit_timer <= {CLKS_PER_BIT{1'b0}};
      end else begin
        bit_timer <= bit_timer - 1;
      end

      if (bit_cnt == 5'd9) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
    end
  end

endmodule
```

### `qwen3_v2.v`

```verilog
module uart_tx #(
  parameter CLKS_PER_BIT = 16
) (
  input  wire       clk,
  input  wire       rst,      // active-high, synchronous
  input  wire [7:0] data_in,
  input  wire       start,    // 1-cycle pulse: send data_in
  output reg        tx,
  output reg        busy,
  output reg        done
);

  reg [7:0] data_reg;
  reg [4:0] bit_cnt;
  reg [CLKS_PER_BIT-1:0] bit_timer;

  always @(posedge clk) begin
    if (rst) begin
      data_reg <= 8'd0;
      bit_cnt <= 5'd0;
      bit_timer <= {CLKS_PER_BIT{1'b0}};
      tx <= 1'b1;
      busy <= 1'b0;
      done <= 1'b0;
    end else if (start && !busy) begin
      data_reg <= data_in;
      bit_cnt <= 5'd1;
      bit_timer <= CLKS_PER_BIT-1;
      tx <= 1'b0;
      busy <= 1'b1;
      done <= 1'b0;
    end else if (busy) begin
      if (bit_timer == 0) begin
        if (bit_cnt == 5'd0) begin
          // Start bit
          tx <= 1'b0;
        end else if (bit_cnt < 5'd8) begin
          // Data bits (LSB first)
          tx <= data_reg[bit_cnt - 1];
        end else begin
          // Stop bit
          tx <= 1'b1;
        end
        bit_cnt <= bit_cnt + 1;
        bit_timer <= CLKS_PER_BIT-1;
      end else begin
        bit_timer <= bit_timer - 1;
      end

      if (bit_cnt == 5'd9) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
    end
  end

endmodule
```

### `qwen3_v3.v`

```verilog
module uart_tx #(
  parameter CLKS_PER_BIT = 16
) (
  input  wire       clk,
  input  wire       rst,      // active-high, synchronous
  input  wire [7:0] data_in,
  input  wire       start,    // 1-cycle pulse: send data_in
  output reg        tx,
  output reg        busy,
  output reg        done
);

  reg [7:0] data_reg;
  reg [4:0] bit_cnt;
  reg [CLKS_PER_BIT-1:0] bit_timer;

  always @(posedge clk) begin
    if (rst) begin
      data_reg <= 8'd0;
      bit_cnt <= 5'd0;
      bit_timer <= {CLKS_PER_BIT{1'b0}};
      tx <= 1'b1;
      busy <= 1'b0;
      done <= 1'b0;
    end else if (start && !busy) begin
      data_reg <= data_in;
      bit_cnt <= 5'd1;
      bit_timer <= CLKS_PER_BIT-1;
      tx <= 1'b0;
      busy <= 1'b1;
      done <= 1'b0;
    end else if (busy) begin
      if (bit_timer == 0) begin
        if (bit_cnt == 5'd0) begin
          // Start bit
          tx <= 1'b0;
        end else if (bit_cnt < 5'd9) begin
          // Data bits (LSB first)
          tx <= data_reg[bit_cnt - 1];
        end else begin
          // Stop bit
          tx <= 1'b1;
        end
        bit_cnt <= bit_cnt + 1;
        bit_timer <= CLKS_PER_BIT-1;
      end else begin
        bit_timer <= bit_timer - 1;
      end

      if (bit_cnt == 5'd9) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
    end
  end

endmodule
```

### `qwen3_v4.v`

```verilog
module uart_tx #(
  parameter CLKS_PER_BIT = 16
) (
  input  wire       clk,
  input  wire       rst,      // active-high, synchronous
  input  wire [7:0] data_in,
  input  wire       start,    // 1-cycle pulse: send data_in
  output reg        tx,
  output reg        busy,
  output reg        done
);

  reg [7:0] data_reg;
  reg [4:0] bit_cnt;
  reg [CLKS_PER_BIT-1:0] bit_timer;

  always @(posedge clk) begin
    if (rst) begin
      data_reg <= 8'd0;
      bit_cnt <= 5'd0;
      bit_timer <= {CLKS_PER_BIT{1'b0}};
      tx <= 1'b1;
      busy <= 1'b0;
      done <= 1'b0;
    end else if (start && !busy) begin
      data_reg <= data_in;
      bit_cnt <= 5'd1;
      bit_timer <= CLKS_PER_BIT-1;
      tx <= 1'b0;
      busy <= 1'b1;
      done <= 1'b0;
    end else if (busy) begin
      if (bit_timer == 0) begin
        if (bit_cnt == 5'd0) begin
          // Start bit
          tx <= 1'b0;
        end else if (bit_cnt < 5'd9) begin
          // Data bits (LSB first)
          tx <= data_reg[bit_cnt - 1];
        end else begin
          // Stop bit
          tx <= 1'b1;
        end
        bit_cnt <= bit_cnt + 1;
        bit_timer <= CLKS_PER_BIT-1;
      end else begin
        bit_timer <= bit_timer - 1;
      end

      if (bit_cnt == 5'd10) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
    end
  end

endmodule
```

### `qwen3_v5.v`

```verilog
module uart_tx #(
  parameter CLKS_PER_BIT = 16
) (
  input  wire       clk,
  input  wire       rst,      // active-high, synchronous
  input  wire [7:0] data_in,
  input  wire       start,    // 1-cycle pulse: send data_in
  output reg        tx,
  output reg        busy,
  output reg        done
);

  reg [7:0] data_reg;
  reg [4:0] bit_cnt;
  reg [CLKS_PER_BIT-1:0] bit_timer;

  always @(posedge clk) begin
    if (rst) begin
      data_reg <= 8'd0;
      bit_cnt <= 5'd0;
      bit_timer <= {CLKS_PER_BIT{1'b0}};
      tx <= 1'b1;
      busy <= 1'b0;
      done <= 1'b0;
    end else begin done <= 1'b0; if (start && !busy) begin
      data_reg <= data_in;
      bit_cnt <= 5'd1;
      bit_timer <= CLKS_PER_BIT-1;
      tx <= 1'b0;
      busy <= 1'b1;
      done <= 1'b0;
    end else if (busy) begin
      if (bit_timer == 0) begin
        if (bit_cnt == 5'd0) begin
          // Start bit
          tx <= 1'b0;
        end else if (bit_cnt < 5'd9) begin
          // Data bits (LSB first)
          tx <= data_reg[bit_cnt - 1];
        end else begin
          // Stop bit
          tx <= 1'b1;
        end
        bit_cnt <= bit_cnt + 1;
        bit_timer <= CLKS_PER_BIT-1;
      end else begin
        bit_timer <= bit_timer - 1;
      end

      if (bit_cnt == 5'd10) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
    end
  end
  end

endmodule
```

### `qwen3_double_start.v`

```verilog
module uart_tx #(
  parameter CLKS_PER_BIT = 16
) (
  input  wire       clk,
  input  wire       rst,      // active-high, synchronous
  input  wire [7:0] data_in,
  input  wire       start,    // 1-cycle pulse: send data_in
  output reg        tx,
  output reg        busy,
  output reg        done
);

  reg [7:0] data_reg;
  reg [4:0] bit_cnt;
  reg [CLKS_PER_BIT-1:0] bit_timer;

  always @(posedge clk) begin
    if (rst) begin
      data_reg <= 8'h00;
      bit_cnt <= 5'd0;
      bit_timer <= CLKS_PER_BIT - 1;
      tx <= 1'b1;
      busy <= 1'b0;
      done <= 1'b0;
    end else if (start && !busy) begin
      data_reg <= data_in;
      bit_cnt <= 5'd0;
      bit_timer <= CLKS_PER_BIT - 1;
      tx <= 1'b0;
      busy <= 1'b1;
      done <= 1'b0;
    end else if (busy) begin
      if (bit_timer == 0) begin
        if (bit_cnt == 5'd0) begin
          // Start bit
          tx <= 1'b0;
        end else if (bit_cnt < 5'd8) begin
          // Data bits (LSB first)
          tx <= data_reg[bit_cnt - 1];
        end else begin
          // Stop bit
          tx <= 1'b1;
        end
        bit_cnt <= bit_cnt + 1;
        bit_timer <= CLKS_PER_BIT - 1;
      end else begin
        bit_timer <= bit_timer - 1;
      end

      if (bit_cnt == 5'd9) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
    end
  end

endmodule
```

### `qwen3_double_start_v2.v`

```verilog
module uart_tx #(
  parameter CLKS_PER_BIT = 16
) (
  input  wire       clk,
  input  wire       rst,      // active-high, synchronous
  input  wire [7:0] data_in,
  input  wire       start,    // 1-cycle pulse: send data_in
  output reg        tx,
  output reg        busy,
  output reg        done
);

  reg [7:0] data_reg;
  reg [4:0] bit_cnt;
  reg [CLKS_PER_BIT-1:0] bit_timer;

  always @(posedge clk) begin
    if (rst) begin
      data_reg <= 8'h00;
      bit_cnt <= 5'd0;
      bit_timer <= CLKS_PER_BIT - 1;
      tx <= 1'b1;
      busy <= 1'b0;
      done <= 1'b0;
    end else if (start && !busy) begin
      data_reg <= data_in;
      bit_cnt <= 5'd0;
      bit_timer <= CLKS_PER_BIT - 1;
      tx <= 1'b0;
      busy <= 1'b1;
      done <= 1'b0;
    end else if (busy) begin
      if (bit_timer == 0) begin
        if (bit_cnt < 5'd8) begin
          // Data bits (LSB first)
          tx <= data_reg[bit_cnt];
        end else begin
          // Stop bit
          tx <= 1'b1;
        end
        bit_cnt <= bit_cnt + 1;
        bit_timer <= CLKS_PER_BIT - 1;
      end else begin
        bit_timer <= bit_timer - 1;
      end

      if (bit_cnt == 5'd9) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
    end
  end

endmodule
```

### `qwen3_double_start_v3.v`

```verilog
module uart_tx #(
  parameter CLKS_PER_BIT = 16
) (
  input  wire       clk,
  input  wire       rst,      // active-high, synchronous
  input  wire [7:0] data_in,
  input  wire       start,    // 1-cycle pulse: send data_in
  output reg        tx,
  output reg        busy,
  output reg        done
);

  reg [7:0] data_reg;
  reg [4:0] bit_cnt;
  reg [CLKS_PER_BIT-1:0] bit_timer;

  always @(posedge clk) begin
    if (rst) begin
      data_reg <= 8'h00;
      bit_cnt <= 5'd0;
      bit_timer <= CLKS_PER_BIT - 1;
      tx <= 1'b1;
      busy <= 1'b0;
      done <= 1'b0;
    end else if (start && !busy) begin
      data_reg <= data_in;
      bit_cnt <= 5'd0;
      bit_timer <= CLKS_PER_BIT - 1;
      tx <= 1'b0;
      busy <= 1'b1;
      done <= 1'b0;
    end else if (busy) begin
      if (bit_timer == 0) begin
        if (bit_cnt < 5'd8) begin
          // Data bits (LSB first)
          tx <= data_reg[bit_cnt];
        end else begin
          // Stop bit
          tx <= 1'b1;
        end
        bit_cnt <= bit_cnt + 1;
        bit_timer <= CLKS_PER_BIT - 1;
      end else begin
        bit_timer <= bit_timer - 1;
      end

      if (bit_cnt == 5'd9 && bit_timer == 0) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
    end
  end

endmodule
```

### `qwen3_busy_never_clears.v`

```verilog
module uart_tx #(
  parameter CLKS_PER_BIT = 16
) (
  input  wire       clk,
  input  wire       rst,      // active-high, synchronous
  input  wire [7:0] data_in,
  input  wire       start,    // 1-cycle pulse: send data_in
  output reg        tx,
  output reg        busy,
  output reg        done
);

  reg [7:0] data_reg;
  reg [4:0] bit_cnt;
  reg [CLKS_PER_BIT-1:0] bit_timer;

  always @(posedge clk) begin
    if (rst) begin
      data_reg <= 8'd0;
      bit_cnt <= 5'd0;
      bit_timer <= CLKS_PER_BIT - 1;
      tx <= 1'b1;
      busy <= 1'b0;
      done <= 1'b0;
    end else if (start && !busy) begin
      data_reg <= data_in;
      bit_cnt <= 5'd0;
      bit_timer <= CLKS_PER_BIT - 1;
      tx <= 1'b0;
      busy <= 1'b1;
      done <= 1'b0;
    end else if (busy) begin
      if (bit_timer == 0) begin
        if (bit_cnt < 8) begin
          tx <= data_reg[bit_cnt];
        end else begin
          tx <= 1'b1;
        end
        bit_cnt <= bit_cnt + 1;
        bit_timer <= CLKS_PER_BIT - 1;
      end else begin
        bit_timer <= bit_timer - 1;
      end

      if (bit_cnt == 8) begin
        done <= 1'b1;
      end
    end
  end

endmodule
```

### `qwen3_busy_never_clears_v2.v`

```verilog
module uart_tx #(
  parameter CLKS_PER_BIT = 16
) (
  input  wire       clk,
  input  wire       rst,      // active-high, synchronous
  input  wire [7:0] data_in,
  input  wire       start,    // 1-cycle pulse: send data_in
  output reg        tx,
  output reg        busy,
  output reg        done
);

  reg [7:0] data_reg;
  reg [4:0] bit_cnt;
  reg [CLKS_PER_BIT-1:0] bit_timer;

  always @(posedge clk) begin
    if (rst) begin
      data_reg <= 8'd0;
      bit_cnt <= 5'd0;
      bit_timer <= CLKS_PER_BIT - 1;
      tx <= 1'b1;
      busy <= 1'b0;
      done <= 1'b0;
    end else if (start && !busy) begin
      data_reg <= data_in;
      bit_cnt <= 5'd0;
      bit_timer <= CLKS_PER_BIT - 1;
      tx <= 1'b0;
      busy <= 1'b1;
      done <= 1'b0;
    end else if (busy) begin
      if (bit_timer == 0) begin
        if (bit_cnt < 8) begin
          tx <= data_reg[bit_cnt];
        end else begin
          tx <= 1'b1;
        end
        bit_cnt <= bit_cnt + 1;
        bit_timer <= CLKS_PER_BIT - 1;
      end else begin
        bit_timer <= bit_timer - 1;
      end

      if (bit_cnt == 9 && bit_timer == 0) begin
    busy <= 1'b0;
    done <= 1'b1;
  end
    end
  end

endmodule
```

## Test benches

Each testbench drives the design and records the pins on every clock. It never decides pass or fail; `checker.py` decodes the recorded trace in Python.

### `uart_tx/tb_uart_tx.v`

```verilog
// Golden testbench for uart_tx. Written by the team, never by the model.
//
// It does NOT decide pass/fail itself. It drives the design and records every
// output pin on every clock cycle, then prints the raw traces. checker.py reads
// those traces, decodes the UART line independently, and writes the feedback.
// Keeping the judgement in Python makes the diagnoses easy to extend.
//
// Deliberate traps:
//   * CLKS_PER_BIT is overridden to 12 (the prompt's default is 16), so a
//     design that hard-codes 16 is caught.
//   * data_in is changed to garbage the cycle after start, so a design that
//     reads data_in live instead of latching it is caught.
//   * Bytes are chosen so that bit-reversal is visible (0x41 -> 0x82).

`timescale 1ns/1ps
module tb_uart_tx;
  localparam CPB    = 12;
  localparam NBYTES = 6;
  localparam MAXC   = 6000;

  reg        clk = 0, rst = 1, start = 0;
  reg  [7:0] data_in = 8'h00;
  wire       tx, busy, done;

  uart_tx #(.CLKS_PER_BIT(CPB)) dut (
    .clk(clk), .rst(rst), .data_in(data_in), .start(start),
    .tx(tx), .busy(busy), .done(done)
  );

  always #5 clk = ~clk;

  // ---- trace recording: one sample per cycle, on the falling edge ----
  reg tx_t   [0:MAXC-1];
  reg busy_t [0:MAXC-1];
  reg done_t [0:MAXC-1];
  integer cyc = 0;
  always @(negedge clk) begin
    if (cyc < MAXC) begin
      tx_t[cyc]   <= tx;
      busy_t[cyc] <= busy;
      done_t[cyc] <= done;
    end
    cyc <= cyc + 1;
  end

  reg [7:0] bytes    [0:NBYTES-1];
  integer   start_at [0:NBYTES-1];
  integer   rst_end;
  integer   i, t, n;
  reg       dumped = 0;

  task dump;
    begin
      if (!dumped) begin
        dumped = 1;
        n = (cyc < MAXC) ? cyc : MAXC;
        $display("CPB %0d", CPB);
        $display("RSTEND %0d", rst_end);
        $write("START");
        for (i = 0; i < NBYTES; i = i + 1)
          if (start_at[i] >= 0) $write(" %0d:%02h", start_at[i], bytes[i]);
        $display("");
        $write("TX ");   for (i = 0; i < n; i = i + 1) $write("%b", tx_t[i]);   $display("");
        $write("BUSY "); for (i = 0; i < n; i = i + 1) $write("%b", busy_t[i]); $display("");
        $write("DONE "); for (i = 0; i < n; i = i + 1) $write("%b", done_t[i]); $display("");
        $display("END");
      end
    end
  endtask

  initial begin
    bytes[0] = 8'h41;  // 'A': bit 0 is 1, bit-reversal gives 0x82
    bytes[1] = 8'h96;
    bytes[2] = 8'h0F;
    bytes[3] = 8'hFF;
    bytes[4] = 8'h00;
    bytes[5] = 8'hD2;
    for (i = 0; i < NBYTES; i = i + 1) start_at[i] = -1;
    rst_end = -1;

    rst = 1;
    repeat (5) @(negedge clk);
    rst = 0; rst_end = cyc;
    repeat (4) @(negedge clk);

    for (i = 0; i < NBYTES; i = i + 1) begin
      data_in = bytes[i]; start = 1; start_at[i] = cyc;
      @(negedge clk);
      start = 0; data_in = ~bytes[i];          // trap: must have been latched
      repeat (11 * CPB) @(negedge clk);        // spacing even if busy is broken
      t = 0;
      while (busy !== 1'b0 && t < 40 * CPB) begin @(negedge clk); t = t + 1; end
      repeat (3) @(negedge clk);
    end
    repeat (2 * CPB) @(negedge clk);
    dump;
    $finish;
  end

  // Global timeout in case the design hangs the simulator's event loop.
  initial begin
    #(MAXC * 10);
    dump;
    $finish;
  end
endmodule
```

### `uart_rx/tb_uart_rx.v`

```verilog
// Golden testbench for uart_rx. Written by the team, never by the model.
//
// The testbench generates a correct UART bit stream itself (the golden TX is
// this file, not a model-written uart_tx). It does NOT decide pass/fail: it
// drives rx from a precomputed plan and records data_out / data_valid /
// frame_error on every clock. checker.py reads the traces and writes the
// feedback.
//
// Timing convention: at negedge k the testbench records the outputs into slot
// k and drives rx = plan[k]. The posedge after that sees plan[k], so the
// design's reaction to plan[k] shows up in slot k+1.
//
// Events, in order (kind codes printed in EVENTS):
//   0 N  normal frames with idle gaps:   0x41 0x96 0x0F
//   1 B  back-to-back burst, no gap:     0xFF 0x00 0xD2
//   2 G  a 3-clock low glitch on an idle line (must be ignored)
//   3 E  0xC3 with stop bit = 0, then rx held low 2 more bit-times (a break)
//   4 R  recovery frame after the error: 0xA3
//   5 J  0x55 with bit edges shifted +-2 clocks (only mid-bit sampling survives)
//
// Deliberate traps:
//   * CLKS_PER_BIT is overridden to 12 (the prompt's default is 16).
//   * 0x41 makes bit reversal visible (0x82).
//   * The burst catches receivers that are not back in idle by the end of the stop bit.
//   * The glitch catches receivers that never re-check the start bit.
//   * The break catches receivers that restart on a low line after a frame error.
//   * The jittered frame catches receivers that sample at bit edges, not middles.

`timescale 1ns/1ps
module tb_uart_rx;
  localparam CPB  = 12;
  localparam MAXC = 3000;
  localparam RSTC = 5;
  localparam GAP  = 3 * CPB;

  reg        clk = 0, rst = 1, rx = 1;
  wire [7:0] data_out;
  wire       data_valid, frame_error;

  uart_rx #(.CLKS_PER_BIT(CPB)) dut (
    .clk(clk), .rst(rst), .rx(rx),
    .data_out(data_out), .data_valid(data_valid), .frame_error(frame_error)
  );

  always #5 clk = ~clk;

  reg       plan  [0:MAXC-1];
  reg       dv_t  [0:MAXC-1];
  reg       fe_t  [0:MAXC-1];
  reg [7:0] do_t  [0:MAXC-1];
  integer   ev_kind  [0:15];
  integer   ev_start [0:15];
  reg [7:0] ev_byte  [0:15];
  integer   nev = 0, p = 0, cyc = 0, i, k;

  // ---- record outputs and drive inputs, one slot per clock ----
  always @(negedge clk) begin
    if (cyc < MAXC) begin
      dv_t[cyc] <= data_valid;
      fe_t[cyc] <= frame_error;
      do_t[cyc] <= data_out;
      rx        <= plan[cyc];
      rst       <= (cyc < RSTC);
    end
    cyc <= cyc + 1;
  end

  // ---- plan builders ----
  task put(input v, input integer len);
    integer j;
    begin
      for (j = 0; j < len; j = j + 1) begin plan[p] = v; p = p + 1; end
    end
  endtask

  task mark(input integer kind, input [7:0] b);
    begin
      ev_kind[nev] = kind; ev_start[nev] = p; ev_byte[nev] = b; nev = nev + 1;
    end
  endtask

  // one frame; jit = 1 shifts the bit edges alternately -2 / +2 clocks
  task frame(input integer kind, input [7:0] b, input stopv, input jit);
    integer j;
    begin
      mark(kind, b);
      put(1'b0, jit ? CPB - 2 : CPB);
      for (j = 0; j < 8; j = j + 1)
        put(b[j], jit ? ((j % 2 == 0) ? CPB + 4 : CPB - 4) : CPB);
      put(stopv, jit ? CPB + 2 : CPB);
    end
  endtask

  task dump;
    begin
      $display("CPB %0d", CPB);
      $display("RSTEND %0d", RSTC);
      $write("EVENTS");
      for (i = 0; i < nev; i = i + 1) $write(" %0d:%0d:%02h", ev_kind[i], ev_start[i], ev_byte[i]);
      $display("");
      $write("RX ");   for (i = 0; i < p; i = i + 1) $write("%b", plan[i]); $display("");
      $write("DV ");   for (i = 0; i < p; i = i + 1) $write("%b", dv_t[i]); $display("");
      $write("FE ");   for (i = 0; i < p; i = i + 1) $write("%b", fe_t[i]); $display("");
      $write("DOUT "); for (i = 0; i < p; i = i + 1) $write("%h", do_t[i]); $display("");
      $display("END");
    end
  endtask

  initial begin
    put(1'b1, RSTC + 6);                         // reset, then idle
    frame(0, 8'h41, 1'b1, 0); put(1'b1, GAP);
    frame(0, 8'h96, 1'b1, 0); put(1'b1, GAP);
    frame(0, 8'h0F, 1'b1, 0); put(1'b1, GAP);
    frame(1, 8'hFF, 1'b1, 0);                    // burst: no gaps
    frame(1, 8'h00, 1'b1, 0);
    frame(1, 8'hD2, 1'b1, 0); put(1'b1, GAP);
    mark(2, 8'h00); put(1'b0, 3); put(1'b1, 12 * CPB);   // glitch
    frame(3, 8'hC3, 1'b0, 0);                    // bad stop bit ...
    put(1'b0, 2 * CPB); put(1'b1, 12 * CPB);     // ... then a break, then idle
    frame(4, 8'hA3, 1'b1, 0); put(1'b1, GAP);
    frame(5, 8'h55, 1'b1, 1); put(1'b1, GAP);    // jittered edges
    put(1'b1, 2 * CPB);
    for (k = p; k < MAXC; k = k + 1) plan[k] = 1'b1;

    wait (cyc >= p + 1);
    @(negedge clk);
    dump;
    $finish;
  end
endmodule
```

## Test bench errors

This is the output of `uart_tx/checker.py` for each generated file, run with Icarus Verilog. The checker stops at the first failure and gives one fix instruction.

| File | Score | Failing stage |
|---|---|---|
| `qwen3_reload_zero.v` | 0.3 | timing |
| `qwen3_reload_zero_v2.v` | 0.3 | timing |
| `qwen3_v2.v` | 0.6 | data |
| `qwen3_v3.v` | 0.35 | stop_idle |
| `qwen3_v4.v` | 0.9 | busy |
| `qwen3_v5.v` | 0.9 | busy |
| `qwen3_double_start.v` | 0.467 | data |
| `qwen3_double_start_v2.v` | 0.9 | busy |
| `qwen3_double_start_v3.v` | 0.95 | done |
| `qwen3_busy_never_clears.v` | 0.35 | stop_idle |
| `qwen3_busy_never_clears_v2.v` | 0.95 | done |

### `qwen3_reload_zero.v`: stage `timing`, score 0.3

```text
Your whole frame finished in about 8 clocks because `bit_timer` counts DOWN and triggers the next bit when it reaches 0, but you reload it with 0, so it is 0 again on the very next clock and triggers every clock. Whenever you load `bit_timer` (when the frame starts AND when it reaches 0), load CLKS_PER_BIT-1, not 0. Then each bit lasts 12 clocks. Make this exact edit (outside the reset branch): replace
  bit_timer <= {CLKS_PER_BIT{1'b0}};
with
  bit_timer <= CLKS_PER_BIT - 1;
First byte 0x41, tx one character per clock from the start bit:
expected: 000000000000111111111111000000000000000000000000000000000000000000000000000000000000111111111111000000000000111111111111
yours:    001000001111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
```

### `qwen3_reload_zero_v2.v`: stage `timing`, score 0.3

```text
Your whole frame finished in about 8 clocks because `bit_timer` counts DOWN and triggers the next bit when it reaches 0, but you reload it with 0, so it is 0 again on the very next clock and triggers every clock. Whenever you load `bit_timer` (when the frame starts AND when it reaches 0), load CLKS_PER_BIT-1, not 0. Then each bit lasts 12 clocks. Make this exact edit (outside the reset branch): replace
  bit_timer <= {CLKS_PER_BIT{1'b0}};
with
  bit_timer <= CLKS_PER_BIT - 1;
First byte 0x41, tx one character per clock from the start bit:
expected: 000000000000111111111111000000000000000000000000000000000000000000000000000000000000111111111111000000000000111111111111
yours:    001000001111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
```

### `qwen3_v2.v`: stage `data`, score 0.6

```text
0x41 arrived as 0xC1: data bits 0..6 are right but data bit 7 always reads 1, because the stop bit arrives one slot early. You send only 7 data bits. Send all 8 (bits 0 through 7), then the stop bit. Check the comparison that ends the data phase: it must allow 8 data bits, not 7.
```

### `qwen3_v3.v`: stage `stop_idle`, score 0.35

```text
After byte 0x41 finished, tx was still 0 when the next byte started: the line never returned to idle. After data bit 7, drive tx = 1 for one full bit-time (the stop bit) and keep busy = 1 until it is over; tx must then stay 1 while idle. Check that busy is not cleared before the stop bit is sent.
```

### `qwen3_v4.v`: stage `busy`, score 0.9

```text
busy dropped to 0 as soon as the stop bit STARTED. The stop bit lasts CLKS_PER_BIT clocks too: keep busy = 1 through it, and clear busy (and pulse done) only when the bit timer expires at the END of the stop bit. Your check
  if (bit_cnt == 5'd10) begin
is true on the first clock of the stop bit. Exact edit: change that condition to
  if (bit_cnt == 5'd10 && bit_timer == 0) begin
so busy clears when the stop bit's timer runs out.
```

### `qwen3_v5.v`: stage `busy`, score 0.9

```text
busy dropped to 0 as soon as the stop bit STARTED. The stop bit lasts CLKS_PER_BIT clocks too: keep busy = 1 through it, and clear busy (and pulse done) only when the bit timer expires at the END of the stop bit. Your check
  if (bit_cnt == 5'd10) begin
is true on the first clock of the stop bit. Exact edit: change that condition to
  if (bit_cnt == 5'd10 && bit_timer == 0) begin
so busy clears when the stop bit's timer runs out.
```

### `qwen3_double_start.v`: stage `data`, score 0.467

```text
0x41 arrived as 0x82: you send the start bit TWICE. You already drive tx <= 0 when start is accepted, and then the `bit_cnt == 0` branch sends 0 again for a whole bit-time. So data bit 0 goes out one slot late, and because you send `data_reg[bit_cnt - 1]`, data bit 7 is never sent. Exact edit: delete the `bit_cnt == 0` start-bit branch, and replace
  tx <= data_reg[bit_cnt - 1];
with
  tx <= data_reg[bit_cnt];
for bit_cnt = 0..7 (8 data bits), then send the stop bit (tx <= 1) when bit_cnt == 8.
```

### `qwen3_double_start_v2.v`: stage `busy`, score 0.9

```text
busy dropped to 0 as soon as the stop bit STARTED. The stop bit lasts CLKS_PER_BIT clocks too: keep busy = 1 through it, and clear busy (and pulse done) only when the bit timer expires at the END of the stop bit. Your check
  if (bit_cnt == 5'd9) begin
is true on the first clock of the stop bit. Exact edit: change that condition to
  if (bit_cnt == 5'd9 && bit_timer == 0) begin
so busy clears when the stop bit's timer runs out.
```

### `qwen3_double_start_v3.v`: stage `done`, score 0.95

```text
done must be high for exactly ONE clock per byte. Default it to 0 at the top of the clocked block and set it to 1 only on the cycle the stop bit finishes. Exact edit: add the line
  done <= 1'b0;
as the very first statement inside `always @(posedge clk) begin`, before `if (rst)`. Your existing done <= 1'b1 later in the block overrides it on that one clock, so done falls back to 0 on the next clock.
```

### `qwen3_busy_never_clears.v`: stage `stop_idle`, score 0.35

```text
busy never went back to 0 after byte 0x96, so the next start was ignored and your bit counter kept running past the stop bit, putting data bits on tx again instead of idling at 1. The only busy <= 0 in your code is in the reset branch. Your stop bit itself is fine. Set busy <= 0 (and done <= 1) on the clock the stop bit's timer runs out. Exact edit: replace
  if (bit_cnt == 8) begin
  done <= 1'b1;
  end
with
  if (bit_cnt == 9 && bit_timer == 0) begin
    busy <= 1'b0;
    done <= 1'b1;
  end
```

### `qwen3_busy_never_clears_v2.v`: stage `done`, score 0.95

```text
done must be high for exactly ONE clock per byte. Default it to 0 at the top of the clocked block and set it to 1 only on the cycle the stop bit finishes. Exact edit: add the line
  done <= 1'b0;
as the very first statement inside `always @(posedge clk) begin`, before `if (rst)`. Your existing done <= 1'b1 later in the block overrides it on that one clock, so done falls back to 0 on the next clock.
```
