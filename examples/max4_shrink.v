module max4(input [3:0] a, input [3:0] b, input [3:0] c, input [3:0] d, output [3:0] y);
  wire [3:0] m1 = (a > b) ? a : b;
  wire [3:0] m2 = (a > c) ? a : c;
  wire [3:0] m3 = (b > d) ? b : d;
  wire [3:0] m4 = (c > d) ? c : d;
  wire [3:0] m5 = (m1 > m2) ? m1 : m2;
  wire [3:0] m6 = (m3 > m4) ? m3 : m4;
  assign y = (m5 > m6) ? m5 : m6;
endmodule
