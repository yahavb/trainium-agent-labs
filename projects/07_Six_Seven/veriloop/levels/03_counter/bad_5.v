// a while loop that never exits once en is 1: the simulation hangs
module counter8(input clk, input reset, input en, output reg [7:0] count);
  reg t;
  always @* while (en) t = ~t;
  always @(posedge clk) if (reset) count <= 0; else if (en) count <= count + 1;
endmodule
