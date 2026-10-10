// One roll per sampled low-to-high roll transition; result holds between rolls.
// The checker validates the public output contract, not this PRNG's exact values.
module dice_roller (
    input clk,
    input rst_n,
    input [1:0] die_select,
    input roll,
    output reg [7:0] rolled_number
);
    reg [31:0] random_state;
    reg roll_previous;
    reg [7:0] sides;
    always @* begin
        case (die_select)
            0: sides = 4;
            1: sides = 6;
            2: sides = 8;
            default: sides = 20;
        endcase
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            random_state <= 32'h1ACE_B00C;
            roll_previous <= 0;
            rolled_number <= 0;
        end else begin
            random_state <= {random_state[30:0],
                             random_state[31] ^ random_state[21] ^ random_state[1] ^ random_state[0]};
            roll_previous <= roll;
            if (roll && !roll_previous)
                rolled_number <= (random_state % sides) + 1;
        end
    end
endmodule
