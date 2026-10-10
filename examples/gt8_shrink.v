module gt8(input [7:0] a, input [7:0] b, output gt);
  wire [8:0] d = {1'b0, b} - {1'b0, a};
  wire [8:0] e = {1'b0, a} - {1'b0, b};
  assign gt = d[8] & ~(e[8]) & (a != b);
endmodule
