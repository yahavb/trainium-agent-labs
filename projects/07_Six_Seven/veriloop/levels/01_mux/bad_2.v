// the two select bits are swapped: sel = 1 gives d[2]
module mux4(input [3:0] d, input [1:0] sel, output y);
  assign y = d[{sel[0], sel[1]}];
endmodule
