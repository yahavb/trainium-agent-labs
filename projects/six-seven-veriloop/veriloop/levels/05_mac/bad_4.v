// saturates at the 20-bit limits instead of wrapping around
module mac(input clk, input reset, input clear, input en,
           input signed [7:0] a, input signed [7:0] b,
           output reg signed [19:0] acc);
  wire signed [21:0] next = acc + a * b;
  always @(posedge clk) begin
    if (reset)      acc <= 20'sd0;
    else if (clear) acc <= 20'sd0;
    else if (en)    acc <= (next > 22'sd524287) ? 20'sd524287 : (next < -22'sd524288) ? -20'sd524288 : next[19:0];
  end
endmodule
