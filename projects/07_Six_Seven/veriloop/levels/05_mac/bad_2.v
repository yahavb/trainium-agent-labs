// the 16-bit product is zero-extended instead of sign-extended into the 20-bit sum
module mac(input clk, input reset, input clear, input en,
           input signed [7:0] a, input signed [7:0] b,
           output reg signed [19:0] acc);
  wire signed [15:0] p = a * b;
  always @(posedge clk) begin
    if (reset)      acc <= 20'sd0;
    else if (clear) acc <= 20'sd0;
    else if (en)    acc <= acc + {4'b0000, p};
  end
endmodule
