module lfsr (
    input clk,
    input reset_n,
    output reg [7:0] data
);

    // Initial state
    reg [7:0] data_reg;

    // Always block for reset and clock
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            // Immediate reset: set data to initial state
            data_reg <= 8'b10001010;
        end else begin
            // Compute new bit using XOR of bits 0, 3, 5, and 6
            reg new_bit;
            new_bit = data_reg[0] ^ data_reg[3] ^ data_reg[5] ^ data_reg[6];
            // Shift data to the left, adding new_bit as the least significant bit
            data_reg <= {data_reg[6:0], new_bit};
        end
    end

    // Assign output
    assign data = data_reg;

endmodule
