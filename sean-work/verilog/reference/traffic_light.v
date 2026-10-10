// Enabled edges count toward each light interval; disabled edges hold phase and timer.
module traffic_light_fsm (
    input clk,
    input reset_n,
    input enable,
    output reg red,
    output reg yellow,
    output reg green
);
    reg [1:0] phase;
    reg [5:0] elapsed;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            phase <= 0;
            elapsed <= 0;
            red <= 1;
            yellow <= 0;
            green <= 0;
        end else if (enable) begin
            case (phase)
                0: begin
                    if (elapsed == 31) begin
                        phase <= 1;
                        elapsed <= 0;
                        red <= 0;
                        green <= 1;
                    end else elapsed <= elapsed + 1;
                end
                1: begin
                    if (elapsed == 19) begin
                        phase <= 2;
                        elapsed <= 0;
                        green <= 0;
                        yellow <= 1;
                    end else elapsed <= elapsed + 1;
                end
                2: begin
                    if (elapsed == 6) begin
                        phase <= 0;
                        elapsed <= 0;
                        yellow <= 0;
                        red <= 1;
                    end else elapsed <= elapsed + 1;
                end
                default: begin
                    phase <= 0;
                    elapsed <= 0;
                    red <= 1;
                    yellow <= 0;
                    green <= 0;
                end
            endcase
        end
    end
endmodule
