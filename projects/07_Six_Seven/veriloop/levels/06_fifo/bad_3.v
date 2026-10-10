// dout is a register, so it shows the front byte one cycle late
module fifo4(input clk, input reset, input wr_en, input rd_en, input [7:0] din,
             output [7:0] dout, output full, output empty, output reg [2:0] count);
  reg [7:0] mem [0:3];
  reg [1:0] head, tail;
  reg [7:0] dout_r;
  wire do_wr = wr_en && (count != 3'd4);
  wire do_rd = rd_en && (count != 3'd0);
  always @(posedge clk) begin
    if (reset) begin head <= 2'd0; tail <= 2'd0; count <= 3'd0; dout_r <= 8'd0; end
    else begin
      dout_r <= (count == 3'd0) ? 8'd0 : mem[head];
      if (do_wr) begin mem[tail] <= din; tail <= tail + 2'd1; end
      if (do_rd) head <= head + 2'd1;
      count <= count + do_wr - do_rd;
    end
  end
  assign full  = (count == 3'd4);
  assign empty = (count == 3'd0);
  assign dout  = dout_r;
endmodule
