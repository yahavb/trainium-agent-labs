// multiplies as UNSIGNED numbers: wrong whenever a or b is negative
module mac(input clk, input reset, input clear, input en,
           input signed [7:0] a, input signed [7:0] b,
           output reg signed [19:0] acc);
  always @(posedge clk) begin
    if (reset)      acc <= 20'sd0;
    else if (clear) acc <= 20'sd0;
    else if (en)    acc <= acc + $unsigned(a) * $unsigned(b);
  end
endmodule
