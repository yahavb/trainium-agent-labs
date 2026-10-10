// enable is checked before reset, so reset is ignored while enable is on
module counter8(input clk, input reset, input en, output reg [7:0] count);
  always @(posedge clk)
    if (en) count <= count + 1;
    else if (reset) count <= 0;
endmodule
