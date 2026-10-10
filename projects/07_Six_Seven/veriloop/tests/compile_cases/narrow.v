module add4(input [2:0] a, input [3:0] b, output [3:0] sum, output cout);
  assign {cout, sum} = a + b;
endmodule
