module mac(input clk, input reset, input clear, input en,
           input signed [7:0] a, input signed [7:0] b,
           output reg signed [19:0] acc);
  always @(posedge clk) begin
    if (reset)      acc <= 20'sd0;
    else if (clear) acc <= 20'sd0;
    else if (en)    acc <= acc + a * b;
  end
endmodule
