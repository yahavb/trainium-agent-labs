module mux4(input [3:0] d, input [1:0] sel, output y);
  assign y = d[sel];
endmodule
