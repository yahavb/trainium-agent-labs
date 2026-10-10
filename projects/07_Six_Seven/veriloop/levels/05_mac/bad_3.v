// enable is checked before clear, so clear is ignored while en = 1
module mac(input clk, input reset, input clear, input en,
           input signed [7:0] a, input signed [7:0] b,
           output reg signed [19:0] acc);
  always @(posedge clk) begin
    if (reset)      acc <= 20'sd0;
    else if (en)    acc <= acc + a * b;
    else if (clear) acc <= 20'sd0;
  end
endmodule
