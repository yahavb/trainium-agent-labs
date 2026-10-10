// selects in reverse order: sel = 0 gives d[3]
module mux4(input [3:0] d, input [1:0] sel, output y);
  assign y = d[3 - sel];
endmodule
