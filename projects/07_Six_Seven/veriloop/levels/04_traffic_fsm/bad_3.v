// the button also cuts RED short (it should be ignored outside GREEN)
module traffic_light(input clk, input reset, input ped, output reg [1:0] light, output walk);
  localparam RED = 2'b00, GREEN = 2'b01, YELLOW = 2'b10;
  reg [2:0] shown;
  always @(posedge clk) begin
    if (reset) begin light <= RED; shown <= 3'd1; end
    else if (light == GREEN && ped) begin light <= YELLOW; shown <= 3'd1; end
    else if (light == RED && ped) begin light <= GREEN; shown <= 3'd1; end
    else if ((light == RED && shown < 3) || (light == GREEN && shown < 4) || (light == YELLOW && shown < 2))
      shown <= shown + 3'd1;
    else begin light <= (light == RED) ? GREEN : (light == GREEN) ? YELLOW : RED; shown <= 3'd1; end
  end
  assign walk = (light == RED);
endmodule
