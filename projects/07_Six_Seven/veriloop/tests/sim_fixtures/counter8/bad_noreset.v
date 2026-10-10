// reset is ignored, so count starts undefined
module counter8(input clk, input reset, input en, output reg [7:0] count);
  always @(posedge clk)
    if (en) count <= count + 1;
endmodule
