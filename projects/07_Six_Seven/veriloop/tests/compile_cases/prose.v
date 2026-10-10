Here is the Verilog:
module add4(input [3:0] a, input [3:0] b, output [3:0] sum, output cout);
  assign {cout, sum} = a + b;
endmodule
