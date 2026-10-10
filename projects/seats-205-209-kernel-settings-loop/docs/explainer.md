# The idea in plain terms

For anyone presenting or judging this who has not written a chip kernel. Every number here is in
`docs/note.md`.

## What the setting is

A chip kernel has to move data from the chip's big, slow memory into its small, fast working memory
before it can compute. Three settings in the kernel say how much to move per trip and how much to
keep in working memory for reuse.

Think of a workshop with a warehouse out back and a small workbench:

- **Too little per trip:** you walk to the warehouse constantly and the machine sits idle waiting for parts.
- **Too much per trip:** the bench overflows, or you haul things you use once and waste the trip.
- **The right amount depends on the job:** the size and shape of the parts decide it.

So there is no single right answer. It changes with the shape of the data and with the number
precision.

## Three ways to pick it

| Way | What we measured |
|---|---|
| Use AWS's suggested settings | Fine at the size AWS tuned for (905.5 µs against a best of 904.7 µs). They do not run at the adapter shape or at any of Qwen3-8B's four layer shapes at 512 tokens. |
| Guess at random | Ends within 10% of the best setting in 73% of runs. |
| Our loop: two small models guided by a checker that reads the chip | Ends within 10% of the best setting in 29 of 30 runs with the final checker. With the first checker it was 3 of 30. |

"The best setting" is known, not assumed: we measured every legal setting on the chip.

## How it helps a model

A model like Qwen3-8B spends most of its time in matrix multiplies, 252 of them per pass. Each one is
a kernel with these settings.

- In a model built from Qwen3-8B's feed-forward block, the settings our loop chose made inference
  13.5% faster than the untuned kernel.
- At 16-bit precision, which real models use, the gap between a bad setting and a good one is 3.3x.

## When it becomes a problem

- **When the shape changes.** At a 2048 square, 39 of 75 settings are within 2% of the best. At the
  adapter shape only 4 of 96 are, and the worst is 7.1x slower than the best.
- **When intuition transfers wrongly.** "Bigger blocks are better" is true at 2048 and badly wrong at
  the adapter shape. That rule of thumb is what made our first checker fail.
- **When precision changes.** Settings tuned at 32-bit are 1.7x off the best at 16-bit.
- **When the vendor default does not apply.** See the first row of the table above.

In short: whenever you are away from the one case someone already tuned by hand.

## Where this is useful

For a plain matrix multiply, AWS's compiler already does as well as our tuned kernel. The use cases
are where the compiler has no ready answer and someone has to write a kernel by hand:

- **New operations:** a custom attention variant, a new normalisation, a quantised or fused operation.
- **New shapes on existing kernels:** adapters, unusual layer widths, different batch or token counts.
- **A different precision:** moving a kernel from 32-bit to 16-bit.
- **A new chip generation:** the memory sizes change, so every hand-tuned setting has to be redone.

In each case an engineer currently finds these settings by trial and error. We tested only matrix
multiply; the other operations are where it would pay off, not something we have shown.

## The pitch in four beats

1. **The problem.** A hand-written kernel has settings that decide its speed. Pick wrong and it is up
   to 7x slower, and the right answer changes with every shape and precision. Today a specialist finds
   it by hand.
2. **What we built.** A loop where two small models tune those settings, with every attempt measured
   on the real chip.
3. **What we found.** The model is not what makes it work; the checker is. With the same model and a
   better checker, good runs went from 3 of 30 to 29 of 30. Random guessing gets 73%.
4. **The limits.** Once the checker is that clear, a one-line rule does better than the model (it
   finds the best setting in 71% of runs against the model's 50%), and for a plain matmul the compiler
   already matches us.

The full script with timings is in `docs/pitch.md`.
