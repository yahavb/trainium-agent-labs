// A: shows the first value during reset; each enabled edge advances
module sequence_generator(input clk, input reset_n, input enable, output reg [7:0] data);
  reg [2:0] idx;
  always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin idx <= 3'd1; data <= 8'hAF; end
    else if (enable) begin
      case (idx)
        3'd0: data <= 8'hAF; 3'd1: data <= 8'hBC; 3'd2: data <= 8'hE2; 3'd3: data <= 8'h78;
        3'd4: data <= 8'hFF; 3'd5: data <= 8'hE2; 3'd6: data <= 8'h0B; 3'd7: data <= 8'h8D;
      endcase
      idx <= idx + 3'd1;
    end else idx <= 3'd1;
  end
endmodule
