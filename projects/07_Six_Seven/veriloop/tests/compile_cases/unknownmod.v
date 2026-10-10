module add4(input [3:0] a, input [3:0] b, output [3:0] sum, output cout);
  full_adder fa0(.a(a[0]), .b(b[0]), .s(sum[0]));
endmodule
