module counter8(input logic clk, input logic reset, input logic en, output logic [7:0] count);
  always_ff @(posedge clk) begin if (reset) count <= '0; else if (en) count <= count + 1; end
endmodule
