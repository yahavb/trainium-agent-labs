// cout is never driven, so it is undefined
module add4(input [3:0] a, input [3:0] b, output [3:0] sum, output cout);
  assign sum = a + b;
endmodule
