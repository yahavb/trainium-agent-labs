// case statement misses sel = 3, so y keeps an old value (a latch)
module mux4(input [3:0] d, input [1:0] sel, output reg y);
  always @* begin
    case (sel)
      2'd0: y = d[0];
      2'd1: y = d[1];
      2'd2: y = d[2];
    endcase
  end
endmodule
