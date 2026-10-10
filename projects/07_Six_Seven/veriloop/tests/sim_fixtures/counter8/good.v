module counter8(input clk, input reset, input en, output reg [7:0] count);
  always @(posedge clk)
    if (reset) count <= 0;
    else if (en) count <= count + 1;
endmodule
