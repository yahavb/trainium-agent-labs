module lfsr (
    input clk,
    input reset_n,
    output reg [7:0] data
);

    // Initialize data to the initial state on reset
    always @(posedge clk) begin
        if (!reset_n) begin
            // Reset: set data to initial state immediately
            data <= 8'b10001010;
        end else begin
            // Compute new bit
            reg new_bit;
            new_bit = data[0] ^ data[3] ^ data[5] ^ data[6];
            // Shift data left, new bit becomes the MSB
            data <= {data[6:0], new_bit};
        end
    end

endmodule
