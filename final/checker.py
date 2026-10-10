"""Self-contained Verilog checker, model loop, and feedback tools.

Examples:
  python checker.py --agent --problem lfsr --repeat 1 --rounds 1 --principles
  python checker.py --spec design.md --ref reference.v --tb design_tb.v --dut candidate.v
  python checker.py --run-tb --tb design_tb.v --dut candidate.v
  python checker.py --init

All implementation and default data live in this file. Missing JSON/spec/Verilog
resources are materialized beside it; existing edited resources are retained.
"""
import argparse, contextlib, fnmatch, json, pathlib, sys, types, unittest

ROOT = pathlib.Path(__file__).resolve().parent


KNOWLEDGE = {'*:COMPILE_ERROR': 'A design description must satisfy the language and interface requirements before its '
                    'behavior can be evaluated.',
 '*:X_BEFORE_FIRST_EDGE': 'An output required before the first clock edge must become defined independently '
                          'of a clocked transition.',
 '*:WRONG_BETWEEN_EDGES_IN_RESET': 'A reset specified to act independently of the clock must restore the '
                                   'required state between clock edges.',
 '*:WRONG_AFTER_EDGE_IN_RESET': 'A clock edge occurring during asserted reset must preserve the specified '
                                'reset state.',
 '*:MULTIPLE_DRIVERS': 'Each signal must have one controlling source or an explicitly defined rule for '
                       'resolving contributions from multiple sources.',
 '*:BLOCKING_IN_CLOCKED_LOGIC': 'Clocked state transitions must preserve the specified sampling '
                                'relationships regardless of simulation evaluation order.',
 '*:INIT_ONLY_VALUE': 'Startup initialization cannot substitute for a reset that must restore state during '
                      'operation.',
 '*:NEXT_STATE_WRONG_FIRST_EDGE': 'The first transition after reset must follow the specified transition '
                                  'rule from the reset state and sampled inputs.',
 '*:NEXT_STATE_DIVERGES_LATER': 'Every transition must satisfy the specified relationship between current '
                                'state, sampled inputs, and subsequent state.',
 '*:TIMEOUT': 'An operation required to finish must make progress toward completion within its specified '
              'time allowance.',
 '*:PARTIAL_X': 'Every bit of an output required to be defined must have a known logical value at its '
                'observation time.',
 '*:OUTPUT_STUCK': 'An output must change whenever the specified behavior requires a different value.',
 '*:OUTPUT_INVERTED': 'Each output bit must have the logical polarity required by the specification.',
 '*:BIT_ORDER_REVERSED': 'Each output position must preserve the bit significance and ordering required by '
                         'the specification.',
 '*:SHIFTED_TOWARD_LSB': 'Movement toward lower bit positions and the entering value must together satisfy '
                         'the specified relationship between consecutive states.',
 '*:SHIFTED_TOWARD_MSB': 'Movement toward higher bit positions and the entering value must together satisfy '
                         'the specified relationship between consecutive states.',
 '*:OFF_BY_ONE_VALUE': 'Numeric outputs must match the specified value exactly, including increments, '
                       'decrements, and boundary behavior.',
 '*:ONE_CYCLE_LATE': 'A required output must become visible on its specified cycle without an additional '
                     'cycle of delay.',
 '*:ONE_CYCLE_EARLY': 'A required output must become visible on its specified cycle without advancing its '
                      'appearance.',
 '*:UPPER_BITS_ZERO': 'Higher output bits must retain their specified information and significance '
                      'throughout the full output width.',
 '*:WRONG_WHILE_DISABLED': 'A disabled function must obey its specified inactive behavior for both visible '
                           'outputs and retained state.',
 '*:WRONG_AFTER_REENABLE': 'Reactivation must follow the specified transition from the state retained or '
                           'established during the disabled interval.',
 '*:WRONG_AFTER_WRAP': 'A cyclic transition must preserve the specified successor relationship at the '
                       'boundary and throughout subsequent operation.',
 'shift_register:COMPILE_ERROR': 'A design description must satisfy the language and interface requirements '
                                 'before its behavior can be evaluated.',
 'shift_register:MULTIPLE_DRIVERS': 'Each signal must have one controlling source or an explicitly defined '
                                    'rule for resolving contributions from multiple sources.',
 'shift_register:TIMEOUT': 'An operation required to finish must make progress toward completion within its '
                           'specified time allowance.',
 'shift_register:PARTIAL_X': 'Every bit of an output required to be defined must have a known logical value '
                             'at its observation time.',
 'shift_register:OUTPUT_STUCK': 'An output must change whenever the specified behavior requires a different '
                                'value.',
 'shift_register:OUTPUT_INVERTED': 'Each output bit must have the logical polarity required by the '
                                   'specification.',
 'shift_register:BIT_ORDER_REVERSED': 'Each output position must preserve the bit significance and ordering '
                                      'required by the specification.',
 'shift_register:X_BEFORE_FIRST_EDGE': 'An output required before the first clock edge must become defined '
                                       'independently of a clocked transition.',
 'shift_register:WRONG_BETWEEN_EDGES_IN_RESET': 'A reset specified to act independently of the clock must '
                                                'restore the required state between clock edges.',
 'shift_register:WRONG_AFTER_EDGE_IN_RESET': 'A clock edge occurring during asserted reset must preserve the '
                                             'specified reset state.',
 'shift_register:BLOCKING_IN_CLOCKED_LOGIC': 'Clocked state transitions must preserve the specified sampling '
                                             'relationships regardless of simulation evaluation order.',
 'shift_register:INIT_ONLY_VALUE': 'Startup initialization cannot substitute for a reset that must restore '
                                   'state during operation.',
 'shift_register:NEXT_STATE_WRONG_FIRST_EDGE': 'The first shift after reset must move the reset contents and '
                                               'introduce new information exactly as specified.',
 'shift_register:NEXT_STATE_DIVERGES_LATER': 'Each shift must preserve the specified relationship between '
                                             'stored contents, incoming information, and subsequent '
                                             'contents.',
 'shift_register:ONE_CYCLE_LATE': 'Shifted information must reach its specified output position on the '
                                  'required clock cycle.',
 'shift_register:ONE_CYCLE_EARLY': 'Shifted information must reach its specified output position without '
                                   'advancing beyond the required clock cycle.',
 'shift_register:SHIFTED_TOWARD_LSB': 'Shifting toward lower bit positions must preserve the specified '
                                      'direction, bit movement, and entering value.',
 'shift_register:SHIFTED_TOWARD_MSB': 'Shifting toward higher bit positions must preserve the specified '
                                      'direction, bit movement, and entering value.',
 'shift_register:UPPER_BITS_ZERO': 'Higher output bits must retain their specified information and '
                                   'significance throughout the full output width.',
 'lfsr:COMPILE_ERROR': 'A design description must satisfy the language and interface requirements before its '
                       'behavior can be evaluated.',
 'lfsr:MULTIPLE_DRIVERS': 'Each signal must have one controlling source or an explicitly defined rule for '
                          'resolving contributions from multiple sources.',
 'lfsr:TIMEOUT': 'An operation required to finish must make progress toward completion within its specified '
                 'time allowance.',
 'lfsr:PARTIAL_X': 'Every bit of an output required to be defined must have a known logical value at its '
                   'observation time.',
 'lfsr:OUTPUT_STUCK': 'An output must change whenever the specified behavior requires a different value.',
 'lfsr:OUTPUT_INVERTED': 'Each output bit must have the logical polarity required by the specification.',
 'lfsr:BIT_ORDER_REVERSED': 'Each output position must preserve the bit significance and ordering required '
                            'by the specification.',
 'lfsr:X_BEFORE_FIRST_EDGE': 'An output required before the first clock edge must become defined '
                             'independently of a clocked transition.',
 'lfsr:WRONG_BETWEEN_EDGES_IN_RESET': 'A reset specified to act independently of the clock must restore the '
                                      'required state between clock edges.',
 'lfsr:WRONG_AFTER_EDGE_IN_RESET': 'A clock edge occurring during asserted reset must preserve the specified '
                                   'reset state.',
 'lfsr:BLOCKING_IN_CLOCKED_LOGIC': 'Clocked state transitions must preserve the specified sampling '
                                   'relationships regardless of simulation evaluation order.',
 'lfsr:INIT_ONLY_VALUE': 'A feedback seed required on reset must be restored during operation, independently '
                         'of startup initialization.',
 'lfsr:NEXT_STATE_WRONG_FIRST_EDGE': 'The first feedback transition must follow the specified recurrence '
                                     'from the reset seed.',
 'lfsr:NEXT_STATE_DIVERGES_LATER': 'Every feedback transition must preserve the specified recurrence, '
                                   'including the relationships among tapped bits and the entering bit.',
 'lfsr:ONE_CYCLE_LATE': 'A required output must become visible on its specified cycle without an additional '
                        'cycle of delay.',
 'lfsr:ONE_CYCLE_EARLY': 'A required output must become visible on its specified cycle without advancing its '
                         'appearance.',
 'lfsr:SHIFTED_TOWARD_LSB': 'Feedback movement toward lower bit positions must agree with the specified '
                            'recurrence and entering bit.',
 'lfsr:SHIFTED_TOWARD_MSB': 'Feedback movement toward higher bit positions must agree with the specified '
                            'recurrence and entering bit.',
 'lfsr:UPPER_BITS_ZERO': 'Higher output bits must retain their specified information and significance '
                         'throughout the full output width.',
 'lfsr:WRONG_AFTER_WRAP': 'Returning through a feedback cycle boundary must preserve the specified '
                          'recurrence without an extra or missing transition.',
 'counter:COMPILE_ERROR': 'A design description must satisfy the language and interface requirements before '
                          'its behavior can be evaluated.',
 'counter:MULTIPLE_DRIVERS': 'Each signal must have one controlling source or an explicitly defined rule for '
                             'resolving contributions from multiple sources.',
 'counter:TIMEOUT': 'An operation required to finish must make progress toward completion within its '
                    'specified time allowance.',
 'counter:PARTIAL_X': 'Every bit of an output required to be defined must have a known logical value at its '
                      'observation time.',
 'counter:OUTPUT_STUCK': 'An output must change whenever the specified behavior requires a different value.',
 'counter:OUTPUT_INVERTED': 'Each output bit must have the logical polarity required by the specification.',
 'counter:BIT_ORDER_REVERSED': 'Each output position must preserve the bit significance and ordering '
                               'required by the specification.',
 'counter:X_BEFORE_FIRST_EDGE': 'An output required before the first clock edge must become defined '
                                'independently of a clocked transition.',
 'counter:WRONG_BETWEEN_EDGES_IN_RESET': 'A reset specified to act independently of the clock must restore '
                                         'the required state between clock edges.',
 'counter:WRONG_AFTER_EDGE_IN_RESET': 'A clock edge occurring during asserted reset must preserve the '
                                      'specified reset state.',
 'counter:BLOCKING_IN_CLOCKED_LOGIC': 'Clocked state transitions must preserve the specified sampling '
                                      'relationships regardless of simulation evaluation order.',
 'counter:INIT_ONLY_VALUE': 'Startup initialization cannot substitute for a reset that must restore state '
                            'during operation.',
 'counter:NEXT_STATE_WRONG_FIRST_EDGE': 'The first count after reset must follow the specified counting rule '
                                        'from the reset count and sampled inputs.',
 'counter:NEXT_STATE_DIVERGES_LATER': 'Every count transition must follow the specified counting rule, '
                                      'including changes in direction and boundary conditions when '
                                      'applicable.',
 'counter:ONE_CYCLE_LATE': 'A required output must become visible on its specified cycle without an '
                           'additional cycle of delay.',
 'counter:ONE_CYCLE_EARLY': 'A required output must become visible on its specified cycle without advancing '
                            'its appearance.',
 'counter:OFF_BY_ONE_VALUE': 'A count must equal the specified numeric state exactly, including the reset '
                             'value and boundary transitions.',
 'counter:UPPER_BITS_ZERO': 'All count bits must preserve their specified numeric significance throughout '
                            'the supported counting range.',
 'counter:WRONG_AFTER_WRAP': 'A wrapping count must continue from its specified boundary successor without '
                             'an extra or missing count.',
 'fsm:COMPILE_ERROR': 'A design description must satisfy the language and interface requirements before its '
                      'behavior can be evaluated.',
 'fsm:MULTIPLE_DRIVERS': 'Each signal must have one controlling source or an explicitly defined rule for '
                         'resolving contributions from multiple sources.',
 'fsm:TIMEOUT': 'An operation required to finish must make progress toward completion within its specified '
                'time allowance.',
 'fsm:PARTIAL_X': 'Every bit of an output required to be defined must have a known logical value at its '
                  'observation time.',
 'fsm:OUTPUT_STUCK': 'State machine outputs must change whenever the specified state and input behavior '
                     'requires a different visible value.',
 'fsm:OUTPUT_INVERTED': 'Each output bit must have the logical polarity required by the specification.',
 'fsm:BIT_ORDER_REVERSED': 'Encoded state or output bits must preserve the significance assigned to each '
                           'position by the specification.',
 'fsm:X_BEFORE_FIRST_EDGE': 'An output required before the first clock edge must become defined '
                            'independently of a clocked transition.',
 'fsm:WRONG_BETWEEN_EDGES_IN_RESET': 'A reset specified to act independently of the clock must restore the '
                                     'required state between clock edges.',
 'fsm:WRONG_AFTER_EDGE_IN_RESET': 'A clock edge occurring during asserted reset must preserve the specified '
                                  'reset state.',
 'fsm:BLOCKING_IN_CLOCKED_LOGIC': 'Clocked state transitions must preserve the specified sampling '
                                  'relationships regardless of simulation evaluation order.',
 'fsm:INIT_ONLY_VALUE': 'Startup initialization cannot substitute for a reset that must restore state during '
                        'operation.',
 'fsm:NEXT_STATE_WRONG_FIRST_EDGE': 'The first state transition after reset must follow the specified '
                                    'transition relation from the reset state and sampled inputs.',
 'fsm:NEXT_STATE_DIVERGES_LATER': 'Each state transition and visible output must agree with the specified '
                                  'transition relation and output behavior.',
 'fsm:ONE_CYCLE_LATE': 'A required output must become visible on its specified cycle without an additional '
                       'cycle of delay.',
 'fsm:ONE_CYCLE_EARLY': 'A required output must become visible on its specified cycle without advancing its '
                        'appearance.',
 'sequence_gen:COMPILE_ERROR': 'A design description must satisfy the language and interface requirements '
                               'before its behavior can be evaluated.',
 'sequence_gen:MULTIPLE_DRIVERS': 'Each signal must have one controlling source or an explicitly defined '
                                  'rule for resolving contributions from multiple sources.',
 'sequence_gen:TIMEOUT': 'An operation required to finish must make progress toward completion within its '
                         'specified time allowance.',
 'sequence_gen:PARTIAL_X': 'Every bit of an output required to be defined must have a known logical value at '
                           'its observation time.',
 'sequence_gen:OUTPUT_STUCK': 'The sequence output must change whenever a required advancement selects a '
                              'different item.',
 'sequence_gen:OUTPUT_INVERTED': 'Each output bit must have the logical polarity required by the '
                                 'specification.',
 'sequence_gen:BIT_ORDER_REVERSED': 'Each output position must preserve the bit significance and ordering '
                                    'required by the specification.',
 'sequence_gen:X_BEFORE_FIRST_EDGE': 'An output required before the first clock edge must become defined '
                                     'independently of a clocked transition.',
 'sequence_gen:WRONG_BETWEEN_EDGES_IN_RESET': 'A reset specified to act independently of the clock must '
                                              'restore the required state between clock edges.',
 'sequence_gen:WRONG_AFTER_EDGE_IN_RESET': 'A clock edge occurring during asserted reset must preserve the '
                                           'specified reset state.',
 'sequence_gen:BLOCKING_IN_CLOCKED_LOGIC': 'Clocked state transitions must preserve the specified sampling '
                                           'relationships regardless of simulation evaluation order.',
 'sequence_gen:INIT_ONLY_VALUE': 'Startup initialization cannot substitute for a reset that must restore '
                                 'state during operation.',
 'sequence_gen:NEXT_STATE_WRONG_FIRST_EDGE': 'The first sequence transition after reset must use the '
                                             'specified reset position and advancement rule.',
 'sequence_gen:NEXT_STATE_DIVERGES_LATER': 'Every sequence transition must preserve the specified item order '
                                           'and progression conditions.',
 'sequence_gen:ONE_CYCLE_LATE': 'Each sequence item must appear on its specified advancement cycle without '
                                'an extra cycle of delay.',
 'sequence_gen:ONE_CYCLE_EARLY': 'Each sequence item must appear on its specified advancement cycle without '
                                 'premature advancement.',
 'sequence_gen:UPPER_BITS_ZERO': 'Higher output bits must retain their specified information and '
                                 'significance throughout the full output width.',
 'sequence_gen:WRONG_AFTER_WRAP': 'Crossing the sequence boundary must select the specified next item and '
                                  'preserve the order of subsequent items.',
 'detector:COMPILE_ERROR': 'A design description must satisfy the language and interface requirements before '
                           'its behavior can be evaluated.',
 'detector:MULTIPLE_DRIVERS': 'Each signal must have one controlling source or an explicitly defined rule '
                              'for resolving contributions from multiple sources.',
 'detector:TIMEOUT': 'An operation required to finish must make progress toward completion within its '
                     'specified time allowance.',
 'detector:PARTIAL_X': 'Every required detection indication must have a known logical value at its '
                       'observation time.',
 'detector:OUTPUT_STUCK': 'A detection output must change whenever the specified detection condition changes '
                          'its required value.',
 'detector:OUTPUT_INVERTED': 'The detection indication must use the specified polarity for matching and '
                             'nonmatching conditions.',
 'converter:COMPILE_ERROR': 'A design description must satisfy the language and interface requirements '
                            'before its behavior can be evaluated.',
 'converter:MULTIPLE_DRIVERS': 'Each signal must have one controlling source or an explicitly defined rule '
                               'for resolving contributions from multiple sources.',
 'converter:TIMEOUT': 'An operation required to finish must make progress toward completion within its '
                      'specified time allowance.',
 'converter:PARTIAL_X': 'Every bit of an output required to be defined must have a known logical value at '
                        'its observation time.',
 'converter:OUTPUT_STUCK': 'A converted output must change whenever the specified mapping assigns a '
                           'different representation to the input.',
 'converter:OUTPUT_INVERTED': 'Each output bit must have the logical polarity required by the specification.',
 'converter:BIT_ORDER_REVERSED': 'Converted bits must preserve the significance and ordering of the '
                                 'specified output representation.',
 'converter:OFF_BY_ONE_VALUE': 'Conversion must preserve the exact specified numeric mapping, including '
                               'rounding and range boundaries when applicable.',
 'converter:UPPER_BITS_ZERO': 'A converted result must retain all required higher bits of the specified '
                              'output representation.',
 'has_enable:COMPILE_ERROR': 'A design description must satisfy the language and interface requirements '
                             'before its behavior can be evaluated.',
 'has_enable:MULTIPLE_DRIVERS': 'Each signal must have one controlling source or an explicitly defined rule '
                                'for resolving contributions from multiple sources.',
 'has_enable:TIMEOUT': 'An operation required to finish must make progress toward completion within its '
                       'specified time allowance.',
 'has_enable:PARTIAL_X': 'Every bit of an output required to be defined must have a known logical value at '
                         'its observation time.',
 'has_enable:OUTPUT_STUCK': 'Enable must permit every output change required by the specified active '
                            'behavior.',
 'has_enable:OUTPUT_INVERTED': 'Each output bit must have the logical polarity required by the '
                               'specification.',
 'has_enable:WRONG_WHILE_DISABLED': 'Deasserting enable must preserve or establish exactly the inactive '
                                    'output and state behavior required by the specification.',
 'has_enable:WRONG_AFTER_REENABLE': 'Reasserting enable must resume or restart from the position required by '
                                    'the specified disabled behavior.',
 'combinational:COMPILE_ERROR': 'A design description must satisfy the language and interface requirements '
                                'before its behavior can be evaluated.',
 'combinational:MULTIPLE_DRIVERS': 'Each signal must have one controlling source or an explicitly defined '
                                   'rule for resolving contributions from multiple sources.',
 'combinational:TIMEOUT': 'An operation required to finish must make progress toward completion within its '
                          'specified time allowance.',
 'combinational:PARTIAL_X': 'Every output bit must resolve to its specified logical value for each valid '
                            'input combination.',
 'combinational:OUTPUT_STUCK': 'Each input combination must produce its specified output, including '
                               'combinations that require a change in output value.',
 'combinational:OUTPUT_INVERTED': 'Each output bit must have the logical polarity required by the '
                                  'specification.',
 'combinational:BIT_ORDER_REVERSED': 'Each output position must preserve the bit significance and ordering '
                                     'required by the specification.',
 'combinational:OFF_BY_ONE_VALUE': 'The numeric output for each valid input combination must equal the '
                                   'specified mapping exactly.',
 'combinational:UPPER_BITS_ZERO': 'The full output width must preserve the information specified for each '
                                  'valid input combination.',
 'generic:COMPILE_ERROR': 'A design description must satisfy the language and interface requirements before '
                          'its behavior can be evaluated.',
 'generic:X_BEFORE_FIRST_EDGE': 'An output required before the first clock edge must become defined '
                                'independently of a clocked transition.',
 'generic:WRONG_BETWEEN_EDGES_IN_RESET': 'A reset specified to act independently of the clock must restore '
                                         'the required state between clock edges.',
 'generic:WRONG_AFTER_EDGE_IN_RESET': 'A clock edge occurring during asserted reset must preserve the '
                                      'specified reset state.',
 'generic:MULTIPLE_DRIVERS': 'Each signal must have one controlling source or an explicitly defined rule for '
                             'resolving contributions from multiple sources.',
 'generic:BLOCKING_IN_CLOCKED_LOGIC': 'Clocked state transitions must preserve the specified sampling '
                                      'relationships regardless of simulation evaluation order.',
 'generic:INIT_ONLY_VALUE': 'Startup initialization cannot substitute for a reset that must restore state '
                            'during operation.',
 'generic:NEXT_STATE_WRONG_FIRST_EDGE': 'The first transition after reset must follow the specified '
                                        'transition rule from the reset state and sampled inputs.',
 'generic:NEXT_STATE_DIVERGES_LATER': 'Every transition must satisfy the specified relationship between '
                                      'current state, sampled inputs, and subsequent state.',
 'generic:TIMEOUT': 'An operation required to finish must make progress toward completion within its '
                    'specified time allowance.',
 'generic:PARTIAL_X': 'Every bit of an output required to be defined must have a known logical value at its '
                      'observation time.',
 'generic:OUTPUT_STUCK': 'An output must change whenever the specified behavior requires a different value.',
 'generic:OUTPUT_INVERTED': 'Each output bit must have the logical polarity required by the specification.',
 'generic:BIT_ORDER_REVERSED': 'Each output position must preserve the bit significance and ordering '
                               'required by the specification.',
 'generic:SHIFTED_TOWARD_LSB': 'Movement toward lower bit positions and the entering value must together '
                               'satisfy the specified relationship between consecutive states.',
 'generic:SHIFTED_TOWARD_MSB': 'Movement toward higher bit positions and the entering value must together '
                               'satisfy the specified relationship between consecutive states.',
 'generic:OFF_BY_ONE_VALUE': 'Numeric outputs must match the specified value exactly, including increments, '
                             'decrements, and boundary behavior.',
 'generic:ONE_CYCLE_LATE': 'A required output must become visible on its specified cycle without an '
                           'additional cycle of delay.',
 'generic:ONE_CYCLE_EARLY': 'A required output must become visible on its specified cycle without advancing '
                            'its appearance.',
 'generic:UPPER_BITS_ZERO': 'Higher output bits must retain their specified information and significance '
                            'throughout the full output width.',
 'generic:WRONG_WHILE_DISABLED': 'A disabled function must obey its specified inactive behavior for both '
                                 'visible outputs and retained state.',
 'generic:WRONG_AFTER_REENABLE': 'Reactivation must follow the specified transition from the state retained '
                                 'or established during the disabled interval.',
 'generic:WRONG_AFTER_WRAP': 'A cyclic transition must preserve the specified successor relationship at the '
                             'boundary and throughout subsequent operation.',
 'detector:X_BEFORE_FIRST_EDGE': 'An output required before the first clock edge must become defined '
                                 'independently of a clocked transition.',
 'detector:WRONG_BETWEEN_EDGES_IN_RESET': 'A reset specified to act independently of the clock must restore '
                                          'the required state between clock edges.',
 'detector:WRONG_AFTER_EDGE_IN_RESET': 'A clock edge occurring during asserted reset must preserve the '
                                       'specified reset state.',
 'detector:BLOCKING_IN_CLOCKED_LOGIC': 'Clocked state transitions must preserve the specified sampling '
                                       'relationships regardless of simulation evaluation order.',
 'detector:INIT_ONLY_VALUE': 'Startup initialization cannot substitute for a reset that must restore state '
                             'during operation.',
 'detector:NEXT_STATE_WRONG_FIRST_EDGE': 'The first detection transition after reset must follow the '
                                         'specified history and sampled input relationships.',
 'detector:NEXT_STATE_DIVERGES_LATER': 'Detection history and outputs must follow the specified progression '
                                       'for every sampled input.',
 'detector:ONE_CYCLE_LATE': 'A detection indication must become visible on its specified cycle without an '
                            'extra cycle of delay.',
 'detector:ONE_CYCLE_EARLY': 'A detection indication must become visible on its specified cycle without '
                             'premature assertion.'}

TAXONOMY = {'lookup': {'rule': "For each fired signature (in checker order), try '<category>:<SIGNATURE>' for each "
                    "detected category in priority order, then '*:<SIGNATURE>'. Append at most 2 "
                    "'Principle:' lines per feedback, most specific first, no duplicate text. No hit means "
                    'no hint.',
            'log': 'Record matched keys per round in the jsonl as kb_hits.'},
 'provenance': 'Written by Claude in a chat that had already seen the LFSR and sequence-generator problems. '
               "The lfsr:* and sequence_gen:* entries are therefore not 'untuned'; exclude them "
               '(--exclude-categories lfsr,sequence_gen) for any run claiming the cache was written without '
               'seeing the problem.',
 'categories': [{'name': 'uart',
                 'spec_keywords': ['uart', 'baud', 'asynchronous serial', 'rs-232'],
                 'port_hints': ['tx', 'rx', 'baud']},
                {'name': 'spi',
                 'spec_keywords': ['spi', 'serial peripheral'],
                 'port_hints': ['sclk', 'mosi', 'miso', 'cs_n', 'ss_n']},
                {'name': 'i2c', 'spec_keywords': ['i2c', 'two-wire', 'iic'], 'port_hints': ['sda', 'scl']},
                {'name': 'fifo',
                 'spec_keywords': ['fifo', 'queue', 'first-in'],
                 'port_hints': ['full', 'empty', 'wr_en', 'rd_en']},
                {'name': 'memory',
                 'spec_keywords': ['ram', 'rom', 'memory', 'register file'],
                 'port_hints': ['addr', 'we', 'wdata', 'rdata']},
                {'name': 'handshake',
                 'spec_keywords': ['valid', 'ready', 'axi', 'stream', 'handshake'],
                 'port_hints': ['valid', 'ready']},
                {'name': 'arbiter',
                 'spec_keywords': ['arbiter', 'arbitration', 'grant', 'round-robin'],
                 'port_hints': ['req', 'gnt', 'grant']},
                {'name': 'cdc',
                 'spec_keywords': ['clock domain', 'synchronizer', 'asynchronous fifo'],
                 'port_hints': ['clk_a', 'clk_b', 'wclk', 'rclk']},
                {'name': 'lfsr', 'spec_keywords': ['lfsr', 'linear feedback'], 'port_hints': []},
                {'name': 'sequence_gen',
                 'spec_keywords': ['sequence generator', 'generate an output sequence'],
                 'port_hints': []},
                {'name': 'detector',
                 'spec_keywords': ['sequence detector', 'detect', 'pattern'],
                 'port_hints': ['detected', 'found', 'match']},
                {'name': 'counter',
                 'spec_keywords': ['counter', 'count up', 'count down'],
                 'port_hints': ['count', 'cnt']},
                {'name': 'shift_register',
                 'spec_keywords': ['shift register', 'shift'],
                 'port_hints': ['shift', 'serial_in', 'data_in']},
                {'name': 'converter',
                 'spec_keywords': ['bcd', 'gray', 'convert', 'encoder', 'decoder'],
                 'port_hints': ['bcd', 'gray']},
                {'name': 'arithmetic',
                 'spec_keywords': ['adder', 'alu', 'multiplier', 'subtract', 'arithmetic'],
                 'port_hints': ['carry', 'overflow', 'sum', 'product']},
                {'name': 'timer',
                 'spec_keywords': ['timer', 'traffic light', 'debounce', 'pwm', 'clock divider', 'divide'],
                 'port_hints': ['pwm', 'tick']},
                {'name': 'random',
                 'spec_keywords': ['random', 'dice', 'die '],
                 'port_hints': ['roll', 'rand']},
                {'name': 'edge_detect',
                 'spec_keywords': ['edge detect', 'rising edge detector', 'pulse on change'],
                 'port_hints': ['pulse', 'edge']},
                {'name': 'fsm', 'spec_keywords': ['state machine', 'fsm', 'states'], 'port_hints': ['state']},
                {'name': 'has_enable', 'spec_keywords': ['enable'], 'port_hints': ['enable', 'en', 'ena']},
                {'name': 'combinational',
                 'spec_keywords': ['combinational'],
                 'port_hints': [],
                 'rule': 'also applies when the design has no clock port'}],
 'signatures': {'COMPILE_ERROR': {'source': 'compile',
                                  'implemented': True,
                                  'detect': 'compiler exits non-zero'},
                'PORT_MISMATCH': {'source': 'compile',
                                  'implemented': False,
                                  'detect': 'compile error names an unknown module or port, or port width '
                                            'differs from the testbench'},
                'WIDTH_MISMATCH': {'source': 'lint',
                                   'implemented': False,
                                   'detect': 'iverilog -Wall reports a width/padding warning in the DUT'},
                'IMPLICIT_NET': {'source': 'lint',
                                 'implemented': False,
                                 'detect': 'iverilog -Wall reports an implicit net declaration in the DUT'},
                'MULTIPLE_DRIVERS': {'source': 'static',
                                     'implemented': True,
                                     'detect': 'an output is written in more than one block'},
                'BLOCKING_IN_CLOCKED_LOGIC': {'source': 'static',
                                              'implemented': True,
                                              'detect': 'a clocked block writes a register with an immediate '
                                                        'update'},
                'NONBLOCKING_IN_COMB_LOGIC': {'source': 'static',
                                              'implemented': False,
                                              'detect': 'a block with no edge in its trigger list uses '
                                                        'deferred updates'},
                'INCOMPLETE_SENSITIVITY': {'source': 'static',
                                           'implemented': False,
                                           'detect': "a combinational block's explicit trigger list omits a "
                                                     'signal it reads (wildcard lists are fine)'},
                'LATCH_INFERRED': {'source': 'static',
                                   'implemented': False,
                                   'detect': 'a combinational block writes a signal on some branches but not '
                                             'all, with no prior default'},
                'INIT_ONLY_VALUE': {'source': 'static',
                                    'implemented': True,
                                    'detect': 'a simulation-start block writes a design output'},
                'UNSYNTHESIZABLE_CONSTRUCT': {'source': 'static',
                                              'implemented': False,
                                              'detect': 'the DUT contains # delays, wait statements, or '
                                                        'file/display system tasks'},
                'COMB_LOOP': {'source': 'simulation',
                              'implemented': False,
                              'detect': 'simulation stalls at a fixed time with no stimulus progress'},
                'UNUSED_INPUT': {'source': 'static',
                                 'implemented': False,
                                 'detect': 'an input port name is never read in the DUT body'},
                'UNDRIVEN_OUTPUT': {'source': 'static',
                                    'implemented': False,
                                    'detect': 'an output port is never written, or reads Z in simulation'},
                'X_BEFORE_FIRST_EDGE': {'source': 'simulation',
                                        'implemented': True,
                                        'detect': 'output is X with reset asserted and no clock edge yet'},
                'WRONG_BETWEEN_EDGES_IN_RESET': {'source': 'simulation',
                                                 'implemented': True,
                                                 'detect': 'reset asserted mid-run; output not at reset '
                                                           'value before the next edge (only when async '
                                                           'reset is required)'},
                'WRONG_AFTER_EDGE_IN_RESET': {'source': 'simulation',
                                              'implemented': True,
                                              'detect': 'reset held across an edge; output not at reset '
                                                        'value afterwards'},
                'RESET_POLARITY_INVERTED': {'source': 'simulation',
                                            'implemented': False,
                                            'detect': 'output reaches the reset value only while reset is '
                                                      'deasserted'},
                'PARTIAL_RESET': {'source': 'simulation',
                                  'implemented': False,
                                  'detect': 'output correct during reset but X on the first cycle after '
                                            'release'},
                'WRONG_AFTER_RESET_RELEASE': {'source': 'simulation',
                                              'implemented': False,
                                              'detect': 'first divergence is the first edge after a mid-run '
                                                        'reset release'},
                'NEXT_STATE_WRONG_FIRST_EDGE': {'source': 'simulation',
                                                'implemented': True,
                                                'detect': 'first edge after reset release diverges'},
                'NEXT_STATE_DIVERGES_LATER': {'source': 'simulation',
                                              'implemented': True,
                                              'detect': 'correct for some cycles, then diverges'},
                'PARTIAL_X': {'source': 'simulation',
                              'implemented': True,
                              'detect': 'some output bits X, others defined'},
                'OUTPUT_STUCK': {'source': 'simulation',
                                 'implemented': True,
                                 'detect': 'output constant over many cycles where the reference changes'},
                'OUTPUT_INVERTED': {'source': 'simulation',
                                    'implemented': True,
                                    'detect': 'output equals bitwise inverse of expected'},
                'BIT_ORDER_REVERSED': {'source': 'simulation',
                                       'implemented': True,
                                       'detect': 'output equals expected with bit order reversed'},
                'SHIFTED_TOWARD_LSB': {'source': 'simulation',
                                       'implemented': True,
                                       'detect': "output's upper bits equal previous state's lower bits "
                                                 'moved down one'},
                'SHIFTED_TOWARD_MSB': {'source': 'simulation',
                                       'implemented': True,
                                       'detect': "output's lower bits equal previous state's upper bits "
                                                 'moved up one'},
                'OFF_BY_ONE_VALUE': {'source': 'simulation',
                                     'implemented': True,
                                     'detect': 'output equals expected plus or minus one'},
                'ONE_CYCLE_LATE': {'source': 'simulation',
                                   'implemented': True,
                                   'detect': 'output at cycle n equals expected at n-1'},
                'ONE_CYCLE_EARLY': {'source': 'simulation',
                                    'implemented': True,
                                    'detect': 'output at cycle n equals expected at n+1'},
                'UPPER_BITS_ZERO': {'source': 'simulation',
                                    'implemented': True,
                                    'detect': 'high bits always 0 where expected has ones'},
                'WRONG_WHILE_DISABLED': {'source': 'simulation',
                                         'implemented': False,
                                         'detect': 'first divergence occurs while an enable-like input is '
                                                   'low'},
                'WRONG_AFTER_REENABLE': {'source': 'simulation',
                                         'implemented': False,
                                         'detect': 'first divergence is within 2 cycles after an enable-like '
                                                   'input rises again'},
                'WRONG_AFTER_WRAP': {'source': 'simulation',
                                     'implemented': False,
                                     'detect': 'first divergence is the step after the reference returns to '
                                               'its post-reset value'},
                'WRONG_AT_TERMINAL_VALUE': {'source': 'simulation',
                                            'implemented': False,
                                            'detect': "first divergence at the reference's maximum value"},
                'OUTPUT_GLITCH': {'source': 'simulation',
                                  'implemented': False,
                                  'detect': 'output changes mid-cycle with no clock edge or reset change'},
                'TOGGLES_EVERY_CYCLE': {'source': 'simulation',
                                        'implemented': False,
                                        'detect': 'output alternates between two values every cycle'},
                'PERIOD_WRONG': {'source': 'simulation',
                                 'implemented': False,
                                 'detect': "output repeats with a period different from the reference's"},
                'TIMEOUT': {'source': 'simulation',
                            'implemented': True,
                            'detect': 'simulation exceeds the time limit'},
                'COMB_WRONG_FOR_SOME_INPUTS': {'source': 'simulation',
                                               'implemented': False,
                                               'detect': 'combinational design: some input vectors mismatch, '
                                                         'others match'},
                'COMB_DEPENDS_ON_HISTORY': {'source': 'simulation',
                                            'implemented': False,
                                            'detect': 'same input vector applied twice gives different '
                                                      'outputs'},
                'SIGNEDNESS_WRONG': {'source': 'simulation',
                                     'implemented': False,
                                     'detect': "mismatches only when an operand's top bit is set"},
                'OVERFLOW_UNHANDLED': {'source': 'simulation',
                                       'implemented': False,
                                       'detect': 'mismatches only when the true result exceeds the output '
                                                 'width'},
                'CARRY_WRONG': {'source': 'simulation',
                                'implemented': False,
                                'detect': 'carry/borrow output differs while the sum bits match'},
                'FLAG_WRONG': {'source': 'simulation',
                               'implemented': False,
                               'detect': 'a status-flag output differs while the data output matches'},
                'VALUE_OUT_OF_RANGE': {'source': 'simulation',
                                       'implemented': False,
                                       'detect': 'output value outside the set the reference ever produces'},
                'DISTRIBUTION_SKEWED': {'source': 'simulation',
                                        'implemented': False,
                                        'detect': 'over many samples, value frequencies differ beyond '
                                                  'tolerance'},
                'DIGIT_OUT_OF_RANGE': {'source': 'simulation',
                                       'implemented': False,
                                       'detect': 'a 4-bit digit field of the output exceeds 9'},
                'FSM_ILLEGAL_STATE': {'source': 'simulation',
                                      'implemented': False,
                                      'detect': 'output becomes X or constant after a specific input '
                                                'sequence'},
                'FSM_MISSED_TRANSITION': {'source': 'simulation',
                                          'implemented': False,
                                          'detect': 'divergence follows an input combination not seen '
                                                    'earlier in the run'},
                'FSM_OUTPUT_TIMING': {'source': 'simulation',
                                      'implemented': False,
                                      'detect': 'outputs match a one-cycle shift of the reference only '
                                                'around input changes'},
                'FSM_NO_RECOVERY': {'source': 'simulation',
                                    'implemented': False,
                                    'detect': 'after a completed operation the output never matches again'},
                'DETECT_MISSES_OVERLAP': {'source': 'simulation',
                                          'implemented': False,
                                          'detect': 'reference asserts on an overlapping match where the DUT '
                                                    'does not'},
                'DETECT_FALSE_POSITIVE': {'source': 'simulation',
                                          'implemented': False,
                                          'detect': 'DUT asserts a detect output where the reference does '
                                                    'not'},
                'PULSE_WIDTH_WRONG': {'source': 'simulation',
                                      'implemented': False,
                                      'detect': "high pulse length differs from the reference's"},
                'DUTY_CYCLE_WRONG': {'source': 'simulation',
                                     'implemented': False,
                                     'detect': "high fraction per period differs from the reference's"},
                'DWELL_TIME_WRONG': {'source': 'simulation',
                                     'implemented': False,
                                     'detect': "time spent in an output state differs from the reference's"},
                'EDGE_DETECT_DOUBLE_PULSE': {'source': 'simulation',
                                             'implemented': False,
                                             'detect': 'DUT pulse lasts more than one cycle or repeats for '
                                                       'one input change'},
                'BIT_PERIOD_WRONG': {'source': 'protocol',
                                     'implemented': False,
                                     'detect': 'UART monitor: measured bit width outside tolerance of '
                                               'clk/baud'},
                'NO_START_BIT': {'source': 'protocol',
                                 'implemented': False,
                                 'detect': 'UART monitor: data appears without a preceding low start bit'},
                'STOP_BIT_MISSING': {'source': 'protocol',
                                     'implemented': False,
                                     'detect': 'UART monitor: line not high at the stop-bit sample point'},
                'BYTE_BIT_ORDER_REVERSED': {'source': 'protocol',
                                            'implemented': False,
                                            'detect': 'decoded byte equals expected with bits reversed'},
                'BYTE_SHIFTED_ONE_BIT': {'source': 'protocol',
                                         'implemented': False,
                                         'detect': 'decoded byte equals expected shifted by one bit'},
                'PARITY_WRONG': {'source': 'protocol',
                                 'implemented': False,
                                 'detect': 'parity bit inconsistent with data bits'},
                'IDLE_LEVEL_WRONG': {'source': 'protocol',
                                     'implemented': False,
                                     'detect': 'serial line not at idle level between frames or after reset'},
                'SAMPLING_POINT_WRONG': {'source': 'protocol',
                                         'implemented': False,
                                         'detect': 'RX decodes correctly at nominal baud but fails with '
                                                   'small baud offsets'},
                'FRAMING_ACCEPTED_BAD': {'source': 'protocol',
                                         'implemented': False,
                                         'detect': 'RX reports a byte as valid for a frame with a low stop '
                                                   'bit'},
                'BUSY_FLAG_WRONG': {'source': 'protocol',
                                    'implemented': False,
                                    'detect': 'busy/ready output disagrees with whether a frame is in '
                                              'flight'},
                'SPI_MODE_WRONG': {'source': 'protocol',
                                   'implemented': False,
                                   'detect': 'SPI monitor decodes correctly only under a different '
                                             'CPOL/CPHA'},
                'CHIP_SELECT_WRONG': {'source': 'protocol',
                                      'implemented': False,
                                      'detect': 'select deasserted mid-word, or never asserted during '
                                                'clocking'},
                'I2C_START_STOP_WRONG': {'source': 'protocol',
                                         'implemented': False,
                                         'detect': 'SDA changes while SCL is high outside start/stop'},
                'I2C_ACK_WRONG': {'source': 'protocol',
                                  'implemented': False,
                                  'detect': 'ninth-clock SDA level wrong for the expected ack/nack'},
                'OPEN_DRAIN_VIOLATED': {'source': 'protocol',
                                        'implemented': False,
                                        'detect': 'DUT drives 1 (not Z) on an open-drain line'},
                'HANDSHAKE_VIOLATED': {'source': 'protocol',
                                       'implemented': False,
                                       'detect': 'beats counted on cycles without valid and ready both high'},
                'DATA_UNSTABLE_WHILE_VALID': {'source': 'protocol',
                                              'implemented': False,
                                              'detect': 'data or valid changes while valid=1 and ready=0'},
                'VALID_DEPENDS_ON_READY': {'source': 'protocol',
                                           'implemented': False,
                                           'detect': 'valid never rises while ready is held low'},
                'FIFO_FLAG_WRONG': {'source': 'simulation',
                                    'implemented': False,
                                    'detect': 'full/empty differs from a model occupancy count'},
                'OVERFLOW_ACCEPTED': {'source': 'simulation',
                                      'implemented': False,
                                      'detect': 'a write while full changes stored data or count'},
                'UNDERFLOW_ACCEPTED': {'source': 'simulation',
                                       'implemented': False,
                                       'detect': 'a read while empty changes count or returns data flagged '
                                                 'valid'},
                'FIFO_ORDER_WRONG': {'source': 'simulation',
                                     'implemented': False,
                                     'detect': 'read data order differs from write order'},
                'FIFO_COUNT_WRONG': {'source': 'simulation',
                                     'implemented': False,
                                     'detect': 'count output differs from model occupancy'},
                'READ_LATENCY_WRONG': {'source': 'simulation',
                                       'implemented': False,
                                       'detect': 'read data matches the reference shifted by one cycle'},
                'READ_DURING_WRITE_WRONG': {'source': 'simulation',
                                            'implemented': False,
                                            'detect': 'mismatch only when read and write address match in '
                                                      'one cycle'},
                'WRITE_ENABLE_IGNORED': {'source': 'simulation',
                                         'implemented': False,
                                         'detect': 'contents change on a cycle with write enable low'},
                'BYTE_ENABLE_WRONG': {'source': 'simulation',
                                      'implemented': False,
                                      'detect': 'a lane with its byte enable low changed'},
                'GRANT_NOT_ONEHOT': {'source': 'simulation',
                                     'implemented': False,
                                     'detect': 'more than one grant bit high, or a grant to a non-requester'},
                'STARVATION': {'source': 'simulation',
                               'implemented': False,
                               'detect': 'a continuously requesting input not granted within N cycles'},
                'CDC_UNSYNCHRONIZED': {'source': 'static',
                                       'implemented': False,
                                       'detect': 'a signal from one clock domain read by logic clocked by '
                                                 'another without two stages'},
                'GRAY_CODE_VIOLATED': {'source': 'simulation',
                                       'implemented': False,
                                       'detect': 'a cross-domain pointer changes more than one bit per '
                                                 'step'}}}

PRINCIPLES = [{'id': 'COMPILE_ERROR',
  'principle': 'A hardware description must satisfy the language rules and compilation environment '
               'requirements before its behavior can be simulated.'},
 {'id': 'X_BEFORE_FIRST_EDGE',
  'principle': 'An output required to be defined before the first clock edge must become known without '
               'depending on a clocked state update.'},
 {'id': 'WRONG_BETWEEN_EDGES_IN_RESET',
  'principle': 'When reset is specified to act independently of the clock, affected outputs must settle to '
               'their reset values without waiting for a clock edge.'},
 {'id': 'WRONG_AFTER_EDGE_IN_RESET',
  'principle': 'When reset is specified to govern a clock edge, affected outputs must settle to their reset '
               'values after that edge.'},
 {'id': 'MULTIPLE_DRIVERS',
  'principle': "A signal's value must be determined by a single driver or by defined rules that resolve "
               'contributions from multiple drivers.'},
 {'id': 'BLOCKING_IN_CLOCKED_LOGIC',
  'principle': 'Clocked state transitions must preserve the specified sampling relationships; simulator '
               'execution order must not introduce unintended dependencies between simultaneous updates.'},
 {'id': 'INIT_ONLY_VALUE',
  'principle': 'A value established only at simulation startup does not imply that reset can establish or '
               'restore that value during operation.'},
 {'id': 'NEXT_STATE_WRONG_FIRST_EDGE',
  'principle': 'The first transition after reset must follow the specified transition rules from the reset '
               'state using the inputs sampled for that transition.'},
 {'id': 'NEXT_STATE_DIVERGES_LATER',
  'principle': 'Every state transition must satisfy the specified behavior for the current state and '
               'relevant inputs, including boundary conditions and repeated operation.'},
 {'id': 'TIMEOUT',
  'principle': 'A simulation expected to finish requires a reachable termination condition and event '
               'processing that can progress toward it.'},
 {'id': 'PARTIAL_X',
  'principle': 'Every bit of an output required to be defined must have a known value at the specified '
               'observation time.'},
 {'id': 'OUTPUT_STUCK',
  'principle': 'A required state transition must advance the observable state when the specified next state '
               'differs from the current state.'},
 {'id': 'OUTPUT_INVERTED',
  'principle': 'Output polarity is part of the specification; each bit must represent the specified logical '
               'value.'},
 {'id': 'BIT_ORDER_REVERSED',
  'principle': 'The ordering of bits determines their meaning; each output position must correspond to the '
               'specified bit position.'},
 {'id': 'SHIFTED_TOWARD_LSB',
  'principle': 'Movement toward lower bit positions must agree with the specified relationship between '
               'successive states.'},
 {'id': 'SHIFTED_TOWARD_MSB',
  'principle': 'Movement toward higher bit positions and the newly entered bit must both agree with the '
               'specified state transition.'},
 {'id': 'OFF_BY_ONE_VALUE',
  'principle': 'Numeric state values must follow the specified transition exactly, including increments, '
               'decrements, and boundary values.'},
 {'id': 'ONE_CYCLE_LATE',
  'principle': 'A specified output must be visible on its required cycle; an additional cycle of delay '
               'changes the behavior.'},
 {'id': 'ONE_CYCLE_EARLY',
  'principle': 'A specified output must become visible on its required cycle; advancing it by a cycle '
               'changes the behavior.'},
 {'id': 'UPPER_BITS_ZERO',
  'principle': 'The full specified output width carries information; higher bit positions must preserve '
               'their required values.'}]

EXPERIMENT_MANIFEST = {'version': 'md_generic_v4_dynamic_taxonomy',
 'taxonomy': 'taxonomy.json',
 'knowledge': 'knowledge.json',
 'context_tokens': 8000,
 'max_tokens': 1500,
 'rules_limit': 2,
 'rule_bytes': 1200,
 'cycles': 256,
 'random_cycles': 128,
 'tasks': [{'name': 'lfsr',
            'spec': 'prompts/specs/LFSR.md',
            'tb': 'TestBench/lfsr_tb.v',
            'ref': 'verilog/reference/lfsr.v'},
           {'name': 'sequence_generator',
            'spec': 'prompts/specs/sequence_generator.md',
            'tb': 'TestBench/sequence_generator_tb.v',
            'ref': 'verilog/reference/sequence_generator.v'},
           {'name': 'traffic_light',
            'spec': 'prompts/specs/traffic_light.md',
            'tb': 'TestBench/traffic_light_tb.v',
            'ref': 'verilog/reference/traffic_light.v'},
           {'name': 'dice_roller',
            'spec': 'prompts/specs/dice_roller.md',
            'tb': 'TestBench/dice_roller_tb.v',
            'ref': 'verilog/reference/dice_roller.v'}]}

REFERENCE_METADATA = {'verilog/reference/dice_roller.json': {'async_reset': True,
                                        'spec_addendum': 'die_select 0, 1, 2, 3 select 4, 6, 8, 20 sides '
                                                         'respectively. One roll occurs on a rising clock '
                                                         'edge that samples roll high after it was sampled '
                                                         'low. The result must be between 1 and the selected '
                                                         'side count, inclusive. Hold it until the next '
                                                         'roll. Reset must establish a defined output. Any '
                                                         'random-number algorithm may be used; exact random '
                                                         'results are not compared.',
                                        'comparisons': {'rolled_number': {'kind': 'range',
                                                                          'selector': 'die_select',
                                                                          'ranges': {'0': [1, 4],
                                                                                     '1': [1, 6],
                                                                                     '2': [1, 8],
                                                                                     '3': [1, 20]},
                                                                          'trigger': {'input': 'roll',
                                                                                      'edge': 'rising'},
                                                                          'hold_outside_trigger': True,
                                                                          'reset': 'defined'}}},
 'verilog/reference/lfsr.json': {'async_reset': True,
                                 'spec_addendum': 'Reset takes effect immediately and sets data to '
                                                  "8'b10001010. Each rising edge shifts data toward the "
                                                  'highest bit; the new bit 0 is the XOR of old bits 0, 3, 5 '
                                                  'and 6.'},
 'verilog/reference/sequence_generator.json': {'async_reset': True,
                                               'spec_addendum': 'Use behavior A: reset immediately displays '
                                                                'AF and positions the sequence at AF. The '
                                                                'first enabled edge produces BC. Each '
                                                                'further enabled edge advances one item, '
                                                                'wrapping from 8D to AF. Hold both output '
                                                                'and position while disabled, then continue '
                                                                'at the next item.'},
 'verilog/reference/traffic_light.json': {'async_reset': True,
                                          'spec_addendum': 'Reset immediately selects red and clears elapsed '
                                                           'time. Count enabled rising edges: red lasts 32, '
                                                           'green lasts 20, and yellow lasts 7 before '
                                                           'repeating. Enable low holds the light and '
                                                           'elapsed count. Exactly one light is on.'}}

BUNDLED_TEXT_FILES = {'prompts/specs/LFSR.md': 'I am trying to create a Verilog model for an LFSR. It must meet the following '
                          'specifications:\n'
                          '\t- Inputs:\n'
                          '\t\t- Clock\n'
                          '        - Active-low reset\n'
                          '\t- Outputs:\n'
                          '\t\t- Data (8-bits)\n'
                          '\n'
                          'The initial state should be 10001010, and the taps should be at locations 1, 4, '
                          '6, and 7.\n'
                          '\n'
                          'How would I write a design that meets these specifications?',
 'prompts/specs/abro_state_machine.md': 'I am trying to create a Verilog model for an ABRO state machine. It '
                                        'must meet the following specifications:\n'
                                        '    - Inputs:\n'
                                        '        - Clock\n'
                                        '        - Active-low reset\n'
                                        '        - A\n'
                                        '        - B\n'
                                        '    - Outputs:\n'
                                        '        - O\n'
                                        '        - State\n'
                                        '\n'
                                        'Other than the main output from ABRO machine, it should output the '
                                        'current state of the machine for use in verification.\n'
                                        '\n'
                                        'The states for this state machine should be one-hot encoded.\n'
                                        '\n'
                                        'How would I write a design that meets these specifications?\n',
 'prompts/specs/binary_to_bcd.md': 'I am trying to create a Verilog model for a binary to '
                                   'binary-coded-decimal converter. It must meet the following '
                                   'specifications:\n'
                                   '\t- Inputs:\n'
                                   '\t\t- Binary input (5-bits)\n'
                                   '\t- Outputs:\n'
                                   "\t\t- BCD (8-bits: 4-bits for the 10's place and 4-bits for the 1's "
                                   'place)\n'
                                   '\n'
                                   'How would I write a design that meets these specifications?',
 'prompts/specs/cpu8.md': 'Design a synthesizable 8-bit accumulator CPU core in Verilog.\n'
                          '\n'
                          'Module name: cpu8\n'
                          '\n'
                          'Inputs:\n'
                          '- clk: clock\n'
                          '- reset_n: asynchronous active-low reset\n'
                          '- enable: execute one instruction when high\n'
                          '- instruction[7:0]: instruction supplied externally\n'
                          '\n'
                          'Outputs:\n'
                          '- acc[7:0]: accumulator\n'
                          '- pc[7:0]: address of the instruction to execute next\n'
                          '- data_out[7:0]: output register\n'
                          '- carry: carry from the most recent ADD instruction\n'
                          '- halted: indicates that execution has stopped\n'
                          '\n'
                          'The core executes exactly one instruction on each rising clock edge when\n'
                          'enable is high and halted is low. External logic supplies instruction for\n'
                          'the current pc; the CPU has no internal instruction memory.\n'
                          '\n'
                          'Instruction encoding:\n'
                          '- instruction[7:4] is the opcode.\n'
                          '- instruction[3:0] is an unsigned immediate value, extended to eight bits.\n'
                          '\n'
                          'Instructions:\n'
                          '\n'
                          '| Opcode | Name | Behavior |\n'
                          '|--------|------|----------|\n'
                          '| 0 | NOP | Increment pc. |\n'
                          '| 1 | LDI | Load the immediate into acc; increment pc. |\n'
                          '| 2 | ADD | Add the immediate to the current acc; store the low eight bits in '
                          'acc, set carry to the ninth bit, and increment pc. |\n'
                          '| 3 | XOR | Replace acc with acc XOR the immediate; increment pc. |\n'
                          '| 4 | AND | Replace acc with acc AND the immediate; increment pc. |\n'
                          '| 5 | JMP | Set pc to the immediate. |\n'
                          '| 6 | JZ | If the current acc is zero, set pc to the immediate; otherwise '
                          'increment pc. |\n'
                          '| 7 | OUT | Copy the current acc into data_out; increment pc. |\n'
                          '| 8 | HLT | Set halted high and leave pc unchanged. |\n'
                          '\n'
                          'Opcodes 9 through 15 behave as NOP.\n'
                          '\n'
                          'Additional requirements:\n'
                          '1. Reset immediately clears acc, pc, data_out, carry, and halted.\n'
                          '2. Reset takes priority over enable and halted.\n'
                          '3. While enable is low, every output retains its value.\n'
                          '4. While halted is high, every output retains its value until reset.\n'
                          '5. Only ADD changes carry, except reset.\n'
                          '6. Only OUT changes data_out, except reset.\n'
                          '7. Every instruction preserves state not explicitly changed by its behavior.\n'
                          '8. pc increments wrap from 255 to 0.\n'
                          '9. NOP, OUT, and HLT ignore the immediate field.\n'
                          '10. The first enabled rising edge after reset executes the instruction\n'
                          '    supplied for pc = 0.\n'
                          '11. Outputs change only on rising clock edges or reset assertion.\n'
                          '\n'
                          'Return the complete cpu8 module in one Verilog code block.\n',
 'prompts/specs/dice_roller.md': 'I am trying to create a Verilog model for a simulated dice roller. It must '
                                 'meet the following specifications:\n'
                                 '    - Inputs:\n'
                                 '        - Clock\n'
                                 '        - Active-low reset\n'
                                 '        - Die select (2-bits)\n'
                                 '        - Roll\n'
                                 '    - Outputs:\n'
                                 '        - Rolled number (up to 8-bits)\n'
                                 '\n'
                                 'The design should simulate rolling either a 4-sided, 6-sided, 8-sided, or '
                                 '20-sided die, based on the input die select. It should roll when the roll '
                                 'input goes high and output the random number based on the number of sides '
                                 'of the selected die.\n'
                                 '\n'
                                 'How would I write a design that meets these specifications?',
 'prompts/specs/sequence_detector.md': 'I am trying to create a Verilog model for a sequence detector. It '
                                       'must meet the following specifications:\n'
                                       '\t- Inputs:\n'
                                       '\t\t- Clock\n'
                                       '\t\t- Active-low reset\n'
                                       '\t\t- Data (3 bits)\n'
                                       '\t- Outputs:\n'
                                       '\t\t- Sequence found\n'
                                       '\n'
                                       'While enabled, it should detect the following sequence of binary '
                                       'input values:\n'
                                       '\t- 0b001\n'
                                       '\t- 0b101\n'
                                       '\t- 0b110\n'
                                       '\t- 0b000\n'
                                       '\t- 0b110\n'
                                       '\t- 0b110\n'
                                       '\t- 0b011\n'
                                       '\t- 0b101\n'
                                       '\n'
                                       'How would I write a design that meets these specifications?',
 'prompts/specs/sequence_generator.md': 'I am trying to create a Verilog model for a sequence generator. It '
                                        'must meet the following specifications:\n'
                                        '\t- Inputs:\n'
                                        '\t\t- Clock\n'
                                        '\t\t- Active-low reset\n'
                                        '\t\t- Enable\n'
                                        '\t- Outputs:\n'
                                        '\t\t- Data (8 bits)\n'
                                        '\n'
                                        'While enabled, it should generate an output sequence of the '
                                        'following hexadecimal values and then repeat:\n'
                                        '\t- 0xAF\n'
                                        '\t- 0xBC\n'
                                        '\t- 0xE2\n'
                                        '\t- 0x78\n'
                                        '\t- 0xFF\n'
                                        '\t- 0xE2\n'
                                        '\t- 0x0B\n'
                                        '\t- 0x8D\n'
                                        '\n'
                                        'How would I write a design that meets these specifications?',
 'prompts/specs/shift_register.md': 'I am trying to create a Verilog model for a shift register. It must '
                                    'meet the following specifications:\n'
                                    '\t- Inputs:\n'
                                    '\t\t- Clock\n'
                                    '\t\t- Active-low reset\n'
                                    '\t\t- Data (1 bit)\n'
                                    '\t\t- Shift enable\n'
                                    '\t- Outputs:\n'
                                    '\t\t- Data (8 bits)\n'
                                    '\n'
                                    'How would I write a design that meets these specifications?',
 'prompts/specs/traffic_light.md': 'I am trying to create a Verilog model for a traffic light state machine. '
                                   'It must meet the following specifications:\n'
                                   '    - Inputs:\n'
                                   '        - Clock\n'
                                   '        - Active-low reset\n'
                                   '        - Enable\n'
                                   '    - Outputs:\n'
                                   '        - Red\n'
                                   '        - Yellow\n'
                                   '        - Green\n'
                                   '\n'
                                   'The state machine should reset to a red light, change from red to green '
                                   'after 32 clock cycles, change from green to yellow after 20 clock '
                                   'cycles, and then change from yellow to red after 7 clock cycles.\n'
                                   '\n'
                                   'How would I write a design that meets these specifications?',
 'TestBench/abro_state_machine_tb.v': '`timescale 1ns/1ps\n'
                                      '\n'
                                      'module tb_abro_state_machine();\n'
                                      '    reg clk;\n'
                                      '    reg rst_n;\n'
                                      '    reg A;\n'
                                      '    reg B;\n'
                                      '    wire O;\n'
                                      '    wire [3:0] State;\n'
                                      '\n'
                                      '    // Instantiate the ABRO state machine\n'
                                      '    abro_state_machine uut (\n'
                                      '        .clk(clk),\n'
                                      '        .rst_n(rst_n),\n'
                                      '        .A(A),\n'
                                      '        .B(B),\n'
                                      '        .O(O),\n'
                                      '        .State(State)\n'
                                      '    );\n'
                                      '\n'
                                      '    // Clock generation\n'
                                      '    always begin\n'
                                      '        #5 clk = ~clk;\n'
                                      '    end\n'
                                      '\n'
                                      '    integer i;\n'
                                      '    reg [3:0] expected_state;\n'
                                      '    reg expected_O;\n'
                                      '\n'
                                      '    // Test stimulus and checking\n'
                                      '    initial begin\n'
                                      '        // Initialize signals\n'
                                      '        clk = 0;\n'
                                      '        rst_n = 1;\n'
                                      '        A = 0;\n'
                                      '        B = 0;\n'
                                      '\n'
                                      '        // Test cases\n'
                                      '        for (i = 0; i < 16; i = i + 1) begin\n'
                                      '            // Determine the input and expected output based on the '
                                      'test case index\n'
                                      '            case (i)\n'
                                      '                0: begin\n'
                                      '                    A = 0; B = 0; // Reset\n'
                                      "                    expected_O = 1'b0;\n"
                                      "                    expected_state = 4'b0001;\n"
                                      '                end\n'
                                      '                1: begin\n'
                                      '                    A = 0; B = 1; // A=0, B=1\n'
                                      "                    expected_O = 1'b0;\n"
                                      "                    expected_state = 4'b0001;\n"
                                      '                end\n'
                                      '                2: begin\n'
                                      '                    A = 1; B = 0; // A=1, B=0\n'
                                      "                    expected_O = 1'b0;\n"
                                      "                    expected_state = 4'b0010;\n"
                                      '                end\n'
                                      '                3: begin\n'
                                      '                    A = 1; B = 1; // A=1, B=1\n'
                                      "                    expected_O = 1'b1;\n"
                                      "                    expected_state = 4'b0100;\n"
                                      '                end\n'
                                      '                default: begin\n'
                                      '                    // Repeat the same sequence of inputs for the '
                                      'remaining test cases\n'
                                      '                    A = (i-1) % 2;\n'
                                      '                    B = ((i-1) / 2) % 2;\n'
                                      "                    expected_O = (A && B) ? 1'b1 : 1'b0;\n"
                                      "                    expected_state = (A && !B) ? 4'b0010 : ((!A && B) "
                                      "? 4'b0001 : 4'b0100);\n"
                                      '                end\n'
                                      '            endcase\n'
                                      '\n'
                                      '            // Apply reset\n'
                                      '            if (i == 0) begin\n'
                                      '                rst_n = 0;\n'
                                      '                @(posedge clk);\n'
                                      '                rst_n = 1;\n'
                                      '            end\n'
                                      '\n'
                                      '            @(posedge clk); // Wait for the state change on the '
                                      'rising edge of the clock\n'
                                      '\n'
                                      '            // Check the output and state\n'
                                      '            if (O !== expected_O || State !== expected_state) begin\n'
                                      '                $display("Error: Test case %0d failed (A=%b, B=%b)", '
                                      'i, A, B);\n'
                                      '                $display("  Expected: O=%b, State=%b", expected_O, '
                                      'expected_state);\n'
                                      '                $display("  Got: O=%b, State=%b", O, State);\n'
                                      '                $finish;\n'
                                      '            end\n'
                                      '        end\n'
                                      '\n'
                                      '        $display("All test cases passed.");\n'
                                      '        $finish;\n'
                                      '    end\n'
                                      '\n'
                                      '\n'
                                      '\treg vcd_clk;\n'
                                      'initial begin\n'
                                      '    $dumpfile("my_design.vcd");\n'
                                      '    $dumpvars(0, tb_abro_state_machine);\n'
                                      'end\n'
                                      '\n'
                                      'always #5 vcd_clk = ~vcd_clk; // Toggle clock every 5 time units\n'
                                      '\n'
                                      'endmodule\n'
                                      '\n',
 'TestBench/binary_to_bcd_tb.v': '`timescale 1ns / 1ps\n'
                                 '\n'
                                 'module tb_binary_to_bcd_converter;\n'
                                 '\n'
                                 'reg [4:0] binary_input;\n'
                                 'wire [7:0] bcd_output;\n'
                                 '\n'
                                 'binary_to_bcd_converter uut (\n'
                                 '    .binary_input(binary_input),\n'
                                 '    .bcd_output(bcd_output)\n'
                                 ');\n'
                                 '\n'
                                 'integer i;\n'
                                 'reg [4:0] test_binary;\n'
                                 'reg [7:0] expected_bcd;\n'
                                 '\n'
                                 'initial begin\n'
                                 '    $display("Testing Binary-to-BCD Converter...");\n'
                                 '\n'
                                 '    for (i = 0; i < 32; i++) begin\n'
                                 '        test_binary = i;\n'
                                 '        binary_input = test_binary;\n'
                                 '\n'
                                 '        // Calculate expected BCD output\n'
                                 '        expected_bcd[3:0] = test_binary % 10;\n'
                                 '        expected_bcd[7:4] = test_binary / 10;\n'
                                 '\n'
                                 '        #10; // Wait for the results\n'
                                 '\n'
                                 '        if (bcd_output !== expected_bcd) begin\n'
                                 '            $display("Error: Test case %0d failed. Expected BCD: 8\'b%0b, '
                                 'Got: 8\'b%0b",\n'
                                 '                     test_binary, expected_bcd, bcd_output);\n'
                                 '            $finish;\n'
                                 '        end\n'
                                 '    end\n'
                                 '\n'
                                 '    $display("All test cases passed!");\n'
                                 '    $finish;\n'
                                 'end\n'
                                 '\n'
                                 'reg vcd_clk;\n'
                                 'initial begin\n'
                                 '    $dumpfile("my_design.vcd");\n'
                                 '    $dumpvars(0, tb_binary_to_bcd_converter);\n'
                                 'end\n'
                                 '\n'
                                 'always #5 vcd_clk = ~vcd_clk; // Toggle clock every 5 time units\n'
                                 '\n'
                                 'endmodule\n'
                                 '\n',
 'TestBench/cpu8_tb.v': '`timescale 1ns/1ps\n'
                        '\n'
                        'module tb_cpu8;\n'
                        '    reg clk;\n'
                        '    reg reset_n;\n'
                        '    reg enable;\n'
                        '    reg [7:0] instruction;\n'
                        '    wire [7:0] acc;\n'
                        '    wire [7:0] pc;\n'
                        '    wire [7:0] data_out;\n'
                        '    wire carry;\n'
                        '    wire halted;\n'
                        '\n'
                        '    cpu8 dut (\n'
                        '        .clk(clk),\n'
                        '        .reset_n(reset_n),\n'
                        '        .enable(enable),\n'
                        '        .instruction(instruction),\n'
                        '        .acc(acc),\n'
                        '        .pc(pc),\n'
                        '        .data_out(data_out),\n'
                        '        .carry(carry),\n'
                        '        .halted(halted)\n'
                        '    );\n'
                        '\n'
                        '    reg [7:0] expected_acc;\n'
                        '    reg [7:0] expected_pc;\n'
                        '    reg [7:0] expected_out;\n'
                        '    reg expected_carry;\n'
                        '    reg expected_halted;\n'
                        '    reg [7:0] program_mem [0:15];\n'
                        '    reg [31:0] random_state;\n'
                        '    integer edges;\n'
                        '    integer base_value;\n'
                        '    integer operand;\n'
                        '    integer opcode;\n'
                        '    integer i;\n'
                        '\n'
                        '    task check_state;\n'
                        '        begin\n'
                        '            if (acc !== expected_acc || pc !== expected_pc ||\n'
                        '                data_out !== expected_out || carry !== expected_carry ||\n'
                        '                halted !== expected_halted) begin\n'
                        '                $display("Error: CPU8 state mismatch at checked edge %0d", edges);\n'
                        '                if (acc !== expected_acc) $display("Error: acc mismatch");\n'
                        '                if (pc !== expected_pc) $display("Error: pc mismatch");\n'
                        '                if (data_out !== expected_out) $display("Error: data_out '
                        'mismatch");\n'
                        '                if (carry !== expected_carry) $display("Error: carry mismatch");\n'
                        '                if (halted !== expected_halted) $display("Error: halted '
                        'mismatch");\n'
                        '                $finish;\n'
                        '            end\n'
                        '        end\n'
                        '    endtask\n'
                        '\n'
                        '    // Integer arithmetic in the scoreboard checks eight-bit truncation and carry.\n'
                        '    task advance_model;\n'
                        '        integer immediate;\n'
                        '        integer total;\n'
                        '        begin\n'
                        '            immediate = instruction % 16;\n'
                        '            if (!reset_n) begin\n'
                        '                expected_acc = 0;\n'
                        '                expected_pc = 0;\n'
                        '                expected_out = 0;\n'
                        '                expected_carry = 0;\n'
                        '                expected_halted = 0;\n'
                        '            end else if (enable && !expected_halted) begin\n'
                        '                case (instruction / 16)\n'
                        '                    1: begin\n'
                        '                        expected_acc = immediate;\n'
                        '                        expected_pc = (expected_pc + 1) % 256;\n'
                        '                    end\n'
                        '                    2: begin\n'
                        '                        total = expected_acc + immediate;\n'
                        '                        expected_acc = total % 256;\n'
                        '                        expected_carry = total >= 256;\n'
                        '                        expected_pc = (expected_pc + 1) % 256;\n'
                        '                    end\n'
                        '                    3: begin\n'
                        '                        expected_acc = expected_acc ^ immediate;\n'
                        '                        expected_pc = (expected_pc + 1) % 256;\n'
                        '                    end\n'
                        '                    4: begin\n'
                        '                        expected_acc = expected_acc & immediate;\n'
                        '                        expected_pc = (expected_pc + 1) % 256;\n'
                        '                    end\n'
                        '                    5: expected_pc = immediate;\n'
                        '                    6: begin\n'
                        '                        if (expected_acc == 0) expected_pc = immediate;\n'
                        '                        else expected_pc = (expected_pc + 1) % 256;\n'
                        '                    end\n'
                        '                    7: begin\n'
                        '                        expected_out = expected_acc;\n'
                        '                        expected_pc = (expected_pc + 1) % 256;\n'
                        '                    end\n'
                        '                    8: expected_halted = 1;\n'
                        '                    default: expected_pc = (expected_pc + 1) % 256;\n'
                        '                endcase\n'
                        '            end\n'
                        '        end\n'
                        '    endtask\n'
                        '\n'
                        '    // All stimulus changes happen with clk low; outputs settle for one ns.\n'
                        '    task cycle;\n'
                        '        input [7:0] word;\n'
                        '        input run_enable;\n'
                        '        begin\n'
                        '            if (clk !== 0) begin\n'
                        '                $display("Error: testbench stimulus changed with clk high");\n'
                        '                $finish;\n'
                        '            end\n'
                        '            instruction = word;\n'
                        '            enable = run_enable;\n'
                        '            #2;\n'
                        '            check_state;\n'
                        '            clk = 1;\n'
                        '            edges = edges + 1;\n'
                        '            advance_model;\n'
                        '            #1;\n'
                        '            check_state;\n'
                        '            #1;\n'
                        '            clk = 0;\n'
                        '            #1;\n'
                        '            check_state;\n'
                        '        end\n'
                        '    endtask\n'
                        '\n'
                        '    task apply_reset;\n'
                        '        input run_enable;\n'
                        '        begin\n'
                        "            instruction = 8'h2F;\n"
                        '            enable = run_enable;\n'
                        '            reset_n = 0;\n'
                        '            expected_acc = 0;\n'
                        '            expected_pc = 0;\n'
                        '            expected_out = 0;\n'
                        '            expected_carry = 0;\n'
                        '            expected_halted = 0;\n'
                        '            #1;\n'
                        '            check_state;\n'
                        "            cycle(8'h2F, run_enable);\n"
                        "            cycle(8'h7F, !run_enable);\n"
                        '            reset_n = 1;\n'
                        '            #1;\n'
                        '            check_state;\n'
                        '        end\n'
                        '    endtask\n'
                        '\n'
                        '    task set_acc;\n'
                        '        input integer value;\n'
                        '        integer j;\n'
                        '        begin\n'
                        "            cycle(8'h10, 1);\n"
                        "            for (j = 0; j < value / 15; j = j + 1) cycle(8'h2F, 1);\n"
                        "            cycle(8'h20 | (value % 15), 1);\n"
                        '        end\n'
                        '    endtask\n'
                        '\n'
                        '    initial begin\n'
                        '        clk = 0;\n'
                        '        reset_n = 1;\n'
                        '        enable = 0;\n'
                        '        instruction = 0;\n'
                        '        edges = 0;\n'
                        "        random_state = 32'h7A19C5E3;\n"
                        '        #1;\n'
                        '        apply_reset(0);\n'
                        '\n'
                        '        // First enabled edge, output register, and paused execution.\n'
                        "        cycle(8'h1B, 1);\n"
                        "        cycle(8'h70, 1);\n"
                        "        for (i = 0; i < 5; i = i + 1) cycle(8'h2F, 0);\n"
                        "        cycle(8'h21, 1);\n"
                        "        cycle(8'h70, 1);\n"
                        '\n'
                        '        // Every possible accumulator/immediate pair for ADD, including overflow.\n'
                        '        for (base_value = 0; base_value < 256; base_value = base_value + 1)\n'
                        '            for (operand = 0; operand < 16; operand = operand + 1) begin\n'
                        '                set_acc(base_value);\n'
                        "                cycle(8'h20 | operand, 1);\n"
                        '            end\n'
                        '\n'
                        '        // Every opcode and immediate, with carry already set before execution.\n'
                        '        // Non-ADD instructions must preserve carry; ignored operands are varied.\n'
                        '        for (opcode = 0; opcode < 16; opcode = opcode + 1)\n'
                        '            for (operand = 0; operand < 16; operand = operand + 1) begin\n'
                        '                apply_reset(1);\n'
                        '                set_acc(255);\n'
                        "                cycle(8'h21, 1);\n"
                        "                cycle(8'h15, 1);\n"
                        '                cycle((opcode * 16) | operand, 1);\n'
                        "                cycle(8'h70, 1);\n"
                        '            end\n'
                        '\n'
                        '        // Both outcomes of JZ, with every possible target.\n'
                        '        apply_reset(0);\n'
                        '        for (operand = 0; operand < 16; operand = operand + 1) begin\n'
                        "            cycle(8'h10, 1);\n"
                        "            cycle(8'h60 | operand, 1);\n"
                        "            cycle(8'h11, 1);\n"
                        "            cycle(8'h60 | operand, 1);\n"
                        '        end\n'
                        '\n'
                        '        // Exercise pc rollover through an uninterrupted instruction stream.\n'
                        '        apply_reset(1);\n'
                        "        for (i = 0; i < 300; i = i + 1) cycle(8'h0F, 1);\n"
                        '\n'
                        '        // Halt freezes all state for both enable levels, and reset restarts it.\n'
                        '        set_acc(255);\n'
                        "        cycle(8'h21, 1);\n"
                        "        cycle(8'h70, 1);\n"
                        "        cycle(8'h8F, 1);\n"
                        "        for (i = 0; i < 8; i = i + 1) cycle(8'h2F, i % 2);\n"
                        '        apply_reset(0);\n'
                        "        cycle(8'h1D, 1);\n"
                        "        cycle(8'h70, 1);\n"
                        "        cycle(8'h80, 1);\n"
                        '        apply_reset(1);\n'
                        "        cycle(8'h17, 1);\n"
                        '\n'
                        "        // Fetch a short program using the DUT's current pc.\n"
                        '        for (i = 0; i < 16; i = i + 1) program_mem[i] = 0;\n'
                        "        program_mem[0] = 8'h10;\n"
                        "        program_mem[1] = 8'h63;\n"
                        "        program_mem[2] = 8'h1F;\n"
                        "        program_mem[3] = 8'h15;\n"
                        "        program_mem[4] = 8'h2B;\n"
                        "        program_mem[5] = 8'h70;\n"
                        "        program_mem[6] = 8'h38;\n"
                        "        program_mem[7] = 8'h4F;\n"
                        "        program_mem[8] = 8'h5A;\n"
                        "        program_mem[9] = 8'h10;\n"
                        "        program_mem[10] = 8'h80;\n"
                        '        apply_reset(1);\n'
                        '        for (i = 0; i < 20; i = i + 1) begin\n'
                        '            if (pc < 16) cycle(program_mem[pc], 1);\n'
                        "            else cycle(8'h00, 1);\n"
                        '        end\n'
                        "        if (acc !== 8'd8 || data_out !== 8'd16 || pc !== 8'd10 || halted !== 1'b1) "
                        'begin\n'
                        '            $display("Error: CPU8 program did not reach its specified final '
                        'state");\n'
                        '            $finish;\n'
                        '        end\n'
                        '\n'
                        '        // Seeded instruction/enable changes with regular resets after halt.\n'
                        '        apply_reset(0);\n'
                        '        for (i = 0; i < 512; i = i + 1) begin\n'
                        '            random_state = random_state ^ (random_state << 13);\n'
                        '            random_state = random_state ^ (random_state >> 17);\n'
                        '            random_state = random_state ^ (random_state << 5);\n'
                        '            if (i % 23 == 0) apply_reset(random_state[8]);\n'
                        '            cycle(random_state[7:0], random_state[9]);\n'
                        '        end\n'
                        '\n'
                        '        $display("All test cases passed: CPU8 (%0d checked edges)", edges);\n'
                        '        $finish;\n'
                        '    end\n'
                        '\n'
                        '    initial begin\n'
                        '        #2000000;\n'
                        '        $display("Error: CPU8 testbench timeout");\n'
                        '        $finish;\n'
                        '    end\n'
                        'endmodule\n',
 'TestBench/dice_roller_tb.v': '`timescale 1ns / 1ps\n'
                               '\n'
                               'module tb_dice_roller();\n'
                               '    reg clk;\n'
                               '    reg rst_n;\n'
                               '    reg [1:0] die_select;\n'
                               '    reg roll;\n'
                               '    wire [7:0] rolled_number;\n'
                               '\n'
                               '    dice_roller dut (\n'
                               '        .clk(clk),\n'
                               '        .rst_n(rst_n),\n'
                               '        .die_select(die_select),\n'
                               '        .roll(roll),\n'
                               '        .rolled_number(rolled_number)\n'
                               '    );\n'
                               '\n'
                               '    // Clock generation\n'
                               '    always begin\n'
                               '        #5 clk = ~clk;\n'
                               '    end\n'
                               '\n'
                               '    integer i;\n'
                               '    integer j;\n'
                               '    integer error_count;\n'
                               '    reg [31:0] roll_counts [0:20]; // Declare roll_counts as a memory\n'
                               '\n'
                               '    // Testbench stimulus\n'
                               '    initial begin\n'
                               '\n'
                               '        clk = 0;\n'
                               '        rst_n = 0;\n'
                               '        die_select = 0;\n'
                               '        roll = 0;\n'
                               '\n'
                               '        // Reset and initialization\n'
                               '        #10 rst_n = 1;\n'
                               '        #10 roll = 1;\n'
                               '\n'
                               '        error_count = 0;\n'
                               '\n'
                               '        // Test loop\n'
                               '        for (i = 0; i < 4; i++) begin\n'
                               '            die_select = i;\n'
                               '\n'
                               '            // Clear roll_counts\n'
                               '            for (j = 1; j <= 20; j++) begin\n'
                               '                roll_counts[j] = 0;\n'
                               '            end\n'
                               '\n'
                               '            // Perform 1000 rolls and count the results\n'
                               '            for (j = 0; j < 1000; j++) begin\n'
                               '                #10;\n'
                               '                roll = 0;\n'
                               '                #10;\n'
                               '                roll = 1;\n'
                               '                #10;\n'
                               '                roll = 0;\n'
                               '                #10;\n'
                               '\n'
                               '                // Check the rolled_number is within the expected range\n'
                               '                case (die_select)\n'
                               "                    2'b00: begin\n"
                               '                        if (rolled_number < 1 || rolled_number > 4) begin\n'
                               '                            $display("Error: Invalid roll result for 4-sided '
                               'die: %d", rolled_number);\n'
                               '                            error_count = error_count + 1;\n'
                               '                        end\n'
                               '                    end\n'
                               "                    2'b01: begin\n"
                               '                        if (rolled_number < 1 || rolled_number > 6) begin\n'
                               '                            $display("Error: Invalid roll result for 6-sided '
                               'die: %d", rolled_number);\n'
                               '                            error_count = error_count + 1;\n'
                               '                        end\n'
                               '                    end\n'
                               "                    2'b10: begin\n"
                               '                        if (rolled_number < 1 || rolled_number > 8) begin\n'
                               '                            $display("Error: Invalid roll result for 8-sided '
                               'die: %d", rolled_number);\n'
                               '                            error_count = error_count + 1;\n'
                               '                        end\n'
                               '                    end\n'
                               "                    2'b11: begin\n"
                               '                        if (rolled_number < 1 || rolled_number > 20) begin\n'
                               '                            $display("Error: Invalid roll result for '
                               '20-sided die: %d", rolled_number);\n'
                               '                            error_count = error_count + 1;\n'
                               '                        end\n'
                               '                    end\n'
                               '                endcase\n'
                               '\n'
                               '                roll_counts[rolled_number] = roll_counts[rolled_number] + '
                               '1;\n'
                               '            end\n'
                               '\n'
                               '            $display("Results for die_select %b:", die_select);\n'
                               '            for (j = 1; j <= 20; j++) begin\n'
                               '                if (roll_counts[j] > 0) begin\n'
                               '                    $display("  Rolled %d: %d times", j, roll_counts[j]);\n'
                               '                end\n'
                               '            end\n'
                               '        end\n'
                               '\n'
                               '        if (error_count == 0) begin\n'
                               '            $display("Testbench completed successfully.");\n'
                               '        end else begin\n'
                               '            $display("Testbench completed with %d errors.", error_count);\n'
                               '        end\n'
                               '\n'
                               '        $finish;\n'
                               '    end\n'
                               '\n'
                               'reg vcd_clk;\n'
                               'initial begin\n'
                               '    $dumpfile("my_design.vcd");\n'
                               '    $dumpvars(0, tb_dice_roller);\n'
                               'end\n'
                               '\n'
                               'always #5 vcd_clk = ~vcd_clk; // Toggle clock every 5 time units\n'
                               'endmodule\n'
                               '\n',
 'TestBench/lfsr_tb.v': '`timescale 1ns / 1ps\n'
                        '\n'
                        'module tb_lfsr();\n'
                        '    reg clk;\n'
                        '    reg reset_n;\n'
                        '    wire [7:0] data;\n'
                        '\n'
                        '    // Instantiate the LFSR design\n'
                        '    lfsr lfsr_inst (\n'
                        '        .clk(clk),\n'
                        '        .reset_n(reset_n),\n'
                        '        .data(data)\n'
                        '    );\n'
                        '\n'
                        '    // Clock generation\n'
                        '    always begin\n'
                        '        #5 clk = ~clk;\n'
                        '    end\n'
                        '\n'
                        '    // Variables for stimulus and checking\n'
                        '    integer i;\n'
                        '    reg [7:0] data_expected;\n'
                        '\n'
                        '    // Function to generate the reference LFSR output\n'
                        '    function [7:0] reference_lfsr_output;\n'
                        '        input [7:0] current_state;\n'
                        '        begin\n'
                        '            reference_lfsr_output = {current_state[6:0], current_state[0] ^ '
                        'current_state[3] ^ current_state[5] ^ current_state[6]};\n'
                        '        end\n'
                        '    endfunction\n'
                        '\n'
                        '    // Stimulus and checking\n'
                        '    initial begin\n'
                        '        // Initialize the clock and reset signals\n'
                        '        clk = 0;\n'
                        '        reset_n = 0;\n'
                        '\n'
                        '        // Apply reset and wait for a clock cycle\n'
                        '        #5 reset_n = 1;\n'
                        '\n'
                        '        // Initial state check\n'
                        "        if (data !== 8'b10001010) begin\n"
                        '            $display("Error: At time %t, data = %b, expected = 10001010", $time, '
                        'data);\n'
                        '            $finish;\n'
                        '        end\n'
                        '\n'
                        '        // Set the initial expected state\n'
                        "        data_expected = 8'b10001010;\n"
                        '\n'
                        '        // Run simulation for 256 cycles and check the output\n'
                        '        for (i = 0; i < 256; i = i + 1) begin\n'
                        '            // Update the expected state\n'
                        '            data_expected = reference_lfsr_output(data_expected);\n'
                        '\n'
                        '            // Apply the clock\n'
                        '            #10;\n'
                        '\n'
                        '            // Check the output\n'
                        '            if (data !== data_expected) begin\n'
                        '                $display("Error: At time %t, data = %b, expected = %b", $time, '
                        'data, data_expected);\n'
                        '                $finish;\n'
                        '            end\n'
                        '        end\n'
                        '\n'
                        '        // Simulation successful\n'
                        '        $display("Simulation successful. All test cases passed.");\n'
                        '        $finish;\n'
                        '    end\n'
                        '\n'
                        'endmodule\n'
                        '\n',
 'TestBench/sequence_detector_tb.v': '`timescale 1ns/1ps\n'
                                     '\n'
                                     'module tb_sequence_detector();\n'
                                     '    reg clk;\n'
                                     '    reg reset_n;\n'
                                     '    reg [2:0] data;\n'
                                     '    wire sequence_found;\n'
                                     '\n'
                                     '    // Instantiate the sequence_detector module\n'
                                     '    sequence_detector dut (\n'
                                     '        .clk(clk),\n'
                                     '        .reset_n(reset_n),\n'
                                     '        .data(data),\n'
                                     '        .sequence_found(sequence_found)\n'
                                     '    );\n'
                                     '\n'
                                     '    // Clock generation\n'
                                     '    always begin\n'
                                     '        #5 clk = ~clk;\n'
                                     '    end\n'
                                     '\n'
                                     '    // Test stimulus task\n'
                                     '    task apply_stimulus;\n'
                                     '        input [2:0] data_value;\n'
                                     '        input integer delay_cycles;\n'
                                     '        begin\n'
                                     '            data <= data_value;\n'
                                     '            repeat (delay_cycles) @(posedge clk);\n'
                                     '        end\n'
                                     '    endtask\n'
                                     '\n'
                                     '    // Check output task\n'
                                     '    task check_output;\n'
                                     '        input integer cycle;\n'
                                     '        input expected_value;\n'
                                     '        begin\n'
                                     '            if (sequence_found !== expected_value) begin\n'
                                     '                $display("Error: Cycle %0d, Expected: %b, Got: %b", '
                                     'cycle, expected_value, sequence_found);\n'
                                     '                $finish;\n'
                                     '            end\n'
                                     '        end\n'
                                     '    endtask\n'
                                     '\n'
                                     '    // Testbench stimulus and checking\n'
                                     '    initial begin\n'
                                     '        // Initialize signals\n'
                                     '        clk <= 0;\n'
                                     '        reset_n <= 0;\n'
                                     "        data <= 3'b000;\n"
                                     '\n'
                                     '        // Apply reset\n'
                                     '        @(posedge clk);\n'
                                     '        reset_n <= 1;\n'
                                     '\n'
                                     '        // Test case: Correct sequence\n'
                                     "        apply_stimulus(3'b001, 1); check_output(1, 1'b0);\n"
                                     "        apply_stimulus(3'b101, 1); check_output(2, 1'b0);\n"
                                     "        apply_stimulus(3'b110, 1); check_output(3, 1'b0);\n"
                                     "        apply_stimulus(3'b000, 1); check_output(4, 1'b0);\n"
                                     "        apply_stimulus(3'b110, 1); check_output(5, 1'b0);\n"
                                     "        apply_stimulus(3'b110, 1); check_output(6, 1'b0);\n"
                                     "        apply_stimulus(3'b011, 1); check_output(7, 1'b0);\n"
                                     "        apply_stimulus(3'b101, 1); check_output(8, 1'b1);\n"
                                     '\n'
                                     '        // Test case: Incorrect sequence\n'
                                     "        apply_stimulus(3'b001, 1); check_output(9, 1'b0);\n"
                                     "        apply_stimulus(3'b101, 1); check_output(10, 1'b0);\n"
                                     "        apply_stimulus(3'b010, 1); check_output(11, 1'b0);\n"
                                     "        apply_stimulus(3'b000, 1); check_output(12, 1'b0);\n"
                                     '\n'
                                     '        // Indicate successful test completion\n'
                                     '        $display("All test cases passed.");\n'
                                     '        $finish;\n'
                                     '    end\n'
                                     '\n'
                                     'endmodule\n'
                                     '\n',
 'TestBench/sequence_generator_tb.v': '`timescale 1ns/1ps\n'
                                      '\n'
                                      'module tb_sequence_generator();\n'
                                      '    reg clk;\n'
                                      '    reg reset_n;\n'
                                      '    reg enable;\n'
                                      '    wire [7:0] data;\n'
                                      '\n'
                                      '    // Instantiate the sequence_generator\n'
                                      '    sequence_generator seq_gen (\n'
                                      '        .clk(clk),\n'
                                      '        .reset_n(reset_n),\n'
                                      '        .enable(enable),\n'
                                      '        .data(data)\n'
                                      '    );\n'
                                      '\n'
                                      '    // Clock generation\n'
                                      '    always begin\n'
                                      '        #5 clk = ~clk;\n'
                                      '    end\n'
                                      '\n'
                                      '    // Test sequence\n'
                                      '    localparam TEST_SEQ_LEN = 8;\n'
                                      '    reg [7:0] test_sequence [0:TEST_SEQ_LEN-1];\n'
                                      '\n'
                                      '    // Initialize test sequence array\n'
                                      '    initial begin\n'
                                      "        test_sequence[0] = 8'hAF;\n"
                                      "        test_sequence[1] = 8'hBC;\n"
                                      "        test_sequence[2] = 8'hE2;\n"
                                      "        test_sequence[3] = 8'h78;\n"
                                      "        test_sequence[4] = 8'hFF;\n"
                                      "        test_sequence[5] = 8'hE2;\n"
                                      "        test_sequence[6] = 8'h0B;\n"
                                      "        test_sequence[7] = 8'h8D;\n"
                                      '    end\n'
                                      '\n'
                                      '    integer i;\n'
                                      '\n'
                                      '    // Testbench\n'
                                      '    initial begin\n'
                                      '        // Initialize signals\n'
                                      '        clk = 0;\n'
                                      '        reset_n = 0;\n'
                                      '        enable = 0;\n'
                                      '\n'
                                      '        // Apply reset\n'
                                      '        #10 reset_n = 1;\n'
                                      '        #10 reset_n = 0;\n'
                                      '        #10 reset_n = 1;\n'
                                      '\n'
                                      '        // Test case: sequence generation\n'
                                      '        enable = 1;\n'
                                      '        for (i = 0; i < TEST_SEQ_LEN; i = i + 1) begin\n'
                                      '            @(posedge clk);\n'
                                      '            if (data !== test_sequence[i]) begin\n'
                                      '                $display("Error: Mismatch at position %0d. Expected: '
                                      '%h, Got: %h", i, test_sequence[i], data);\n'
                                      '                $finish;\n'
                                      '            end\n'
                                      '        end\n'
                                      '        $display("Test case passed: sequence generation");\n'
                                      '\n'
                                      '        // Test case: sequence repetition\n'
                                      '        for (i = 0; i < TEST_SEQ_LEN; i = i + 1) begin\n'
                                      '            @(posedge clk);\n'
                                      '            if (data !== test_sequence[i]) begin\n'
                                      '                $display("Error: Mismatch in sequence repetition at '
                                      'position %0d. Expected: %h, Got: %h", i, test_sequence[i], data);\n'
                                      '                $finish;\n'
                                      '            end\n'
                                      '        end\n'
                                      '        $display("Test case passed: sequence repetition");\n'
                                      '\n'
                                      '        // Test case: disable the sequence generator\n'
                                      '        enable = 0;\n'
                                      '        @(posedge clk);\n'
                                      '        if (data !== test_sequence[TEST_SEQ_LEN-1]) begin\n'
                                      '            $display("Error: Sequence generator not disabled. '
                                      'Expected: %h, Got: %h", test_sequence[TEST_SEQ_LEN-1], data);\n'
                                      '            $finish;\n'
                                      '        end\n'
                                      '        $display("Test case passed: disable sequence generator");\n'
                                      '\n'
                                      '        $display("All test cases passed");\n'
                                      '        $finish;\n'
                                      '    end\n'
                                      '\n'
                                      'endmodule\n'
                                      '\n',
 'TestBench/shift_register_tb.v': '`timescale 1ns/1ps\n'
                                  '\n'
                                  'module tb_shift_register();\n'
                                  '    reg clk;\n'
                                  '    reg reset_n;\n'
                                  '    reg data_in;\n'
                                  '    reg shift_enable;\n'
                                  '    wire [7:0] data_out;\n'
                                  '\n'
                                  '    // Instantiate the shift_register\n'
                                  '    shift_register dut (\n'
                                  '        .clk(clk),\n'
                                  '        .reset_n(reset_n),\n'
                                  '        .data_in(data_in),\n'
                                  '        .shift_enable(shift_enable),\n'
                                  '        .data_out(data_out)\n'
                                  '    );\n'
                                  '\n'
                                  '    // Clock generation\n'
                                  '    always begin\n'
                                  '        #5 clk = ~clk;\n'
                                  '    end\n'
                                  '\n'
                                  '    // Test case data\n'
                                  "    reg [7:0] test_case_reset_n = 8'b10111111;\n"
                                  "    reg [7:0] test_case_data_in = 8'b11001011;\n"
                                  "    reg [7:0] test_case_shift_enable = 8'b10111110;\n"
                                  "    reg [63:0] test_case_data_out = 64'h80002850a0408000;\n"
                                  '\n'
                                  '    integer i = 0;\n'
                                  '\n'
                                  '    // Test runner\n'
                                  '    initial begin\n'
                                  '        clk = 0;\n'
                                  '\t\treset_n = 0;\n'
                                  '        data_in = 0;\n'
                                  '        shift_enable = 0;\n'
                                  '\n'
                                  '\t\t@(posedge clk) // Wait for one clock cycle for synchronous reset\n'
                                  '\t\t@(negedge clk) // Starts inputs changing on negedge\n'
                                  '\n'
                                  '        for (i = 0; i < 8; i = i + 1) begin\n'
                                  '            reset_n <= test_case_reset_n[i];\n'
                                  '            data_in <= test_case_data_in[i];\n'
                                  '            shift_enable <= test_case_shift_enable[i];\n'
                                  '\t\t\t@(negedge clk)\n'
                                  '\t\t\t\n'
                                  '            if (data_out !== test_case_data_out[8*i+:8]) begin\n'
                                  '                $display("Error: Test case %0d failed. Expected: %b, Got: '
                                  '%b", i, test_case_data_out[8*i+:8], data_out);\n'
                                  '                $finish;\n'
                                  '            end\n'
                                  '        end\n'
                                  '\n'
                                  '        $display("All test cases passed!");\n'
                                  '        $finish;\n'
                                  '    end\n'
                                  'endmodule\n'
                                  '\n',
 'TestBench/traffic_light_tb.v': '`timescale 1ps/1ps\n'
                                 '\n'
                                 'module tb_traffic_light_fsm();\n'
                                 '\n'
                                 'reg clk;\n'
                                 'reg reset_n;\n'
                                 'reg enable;\n'
                                 'wire red;\n'
                                 'wire yellow;\n'
                                 'wire green;\n'
                                 '\n'
                                 'integer error_count;\n'
                                 '\n'
                                 '// Instantiate the traffic_light_fsm\n'
                                 'traffic_light_fsm traffic_light (\n'
                                 '    .clk(clk),\n'
                                 '    .reset_n(reset_n),\n'
                                 '    .enable(enable),\n'
                                 '    .red(red),\n'
                                 '    .yellow(yellow),\n'
                                 '    .green(green)\n'
                                 ');\n'
                                 '\n'
                                 '// Clock generation\n'
                                 'always begin\n'
                                 '    #5 clk = ~clk;\n'
                                 'end\n'
                                 '\n'
                                 '// Check the output and print test results\n'
                                 'task check_output;\n'
                                 '    input integer cycle;\n'
                                 '    input logic exp_red;\n'
                                 '    input logic exp_yellow;\n'
                                 '    input logic exp_green;\n'
                                 '    input integer test_case;\n'
                                 'begin\n'
                                 '    @(negedge clk);\n'
                                 '    if (red !== exp_red || yellow !== exp_yellow || green !== exp_green) '
                                 'begin\n'
                                 '        $display("Error: Test case %0d failed - FSM did not transition to '
                                 '%s after %0d clock cycles.", test_case, exp_red ? "RED" : (exp_yellow ? '
                                 '"YELLOW" : "GREEN"), cycle);\n'
                                 '        error_count = error_count + 1;\n'
                                 '    end else begin\n'
                                 '        $display("Test case %0d passed - FSM transitioned to %s after %0d '
                                 'clock cycles.", test_case, exp_red ? "RED" : (exp_yellow ? "YELLOW" : '
                                 '"GREEN"), cycle);\n'
                                 '    end\n'
                                 'end\n'
                                 'endtask\n'
                                 '\n'
                                 'initial begin\n'
                                 '    // Initialize signals\n'
                                 '    clk = 0;\n'
                                 '    reset_n = 0;\n'
                                 '    enable = 1;\n'
                                 '\n'
                                 '    // Initialize error_count\n'
                                 '    error_count = 0;\n'
                                 '\n'
                                 '    // Test case 1 - Check if the FSM starts with the RED state\n'
                                 '    check_output(0, 1, 0, 0, 1);\n'
                                 '\n'
                                 '    // Apply reset\n'
                                 '    #5 reset_n = 1;\n'
                                 '\n'
                                 '    // Test case 2 - Check if the FSM transitions to GREEN after 32 clock '
                                 'cycles\n'
                                 '    repeat (32) @(posedge clk);\n'
                                 '    check_output(32, 0, 0, 1, 2);\n'
                                 '\n'
                                 '    // Test case 3 - Check if the FSM transitions to YELLOW after 20 clock '
                                 'cycles\n'
                                 '    repeat (20) @(posedge clk);\n'
                                 '    check_output(20, 0, 1, 0, 3);\n'
                                 '\n'
                                 '    // Test case 4 - Check if the FSM transitions to RED after 7 clock '
                                 'cycles\n'
                                 '    repeat (7) @(posedge clk);\n'
                                 '    check_output(7, 1, 0, 0, 4);\n'
                                 '\n'
                                 '    if (error_count > 0) begin\n'
                                 '        $display("Error: %0d test cases failed.", error_count);\n'
                                 '    end else begin\n'
                                 '        $display("All test cases passed!");\n'
                                 '    end\n'
                                 '    $finish;\n'
                                 'end\n'
                                 '\n'
                                 'endmodule\n'
                                 '\n',
 'verilog/reference/cpu8.v': 'module cpu8 (\n'
                             '    input clk,\n'
                             '    input reset_n,\n'
                             '    input enable,\n'
                             '    input [7:0] instruction,\n'
                             '    output reg [7:0] acc,\n'
                             '    output reg [7:0] pc,\n'
                             '    output reg [7:0] data_out,\n'
                             '    output reg carry,\n'
                             '    output reg halted\n'
                             ');\n'
                             "    wire [7:0] immediate = {4'b0000, instruction[3:0]};\n"
                             "    wire [8:0] addition = {1'b0, acc} + {1'b0, immediate};\n"
                             '\n'
                             '    always @(posedge clk or negedge reset_n) begin\n'
                             '        if (!reset_n) begin\n'
                             "            acc <= 8'd0;\n"
                             "            pc <= 8'd0;\n"
                             "            data_out <= 8'd0;\n"
                             "            carry <= 1'b0;\n"
                             "            halted <= 1'b0;\n"
                             '        end else if (enable && !halted) begin\n'
                             "            pc <= pc + 8'd1;\n"
                             '            case (instruction[7:4])\n'
                             "                4'h1: acc <= immediate;\n"
                             "                4'h2: begin\n"
                             '                    acc <= addition[7:0];\n'
                             '                    carry <= addition[8];\n'
                             '                end\n'
                             "                4'h3: acc <= acc ^ immediate;\n"
                             "                4'h4: acc <= acc & immediate;\n"
                             "                4'h5: pc <= immediate;\n"
                             "                4'h6: if (acc == 8'd0) pc <= immediate;\n"
                             "                4'h7: data_out <= acc;\n"
                             "                4'h8: begin\n"
                             "                    halted <= 1'b1;\n"
                             '                    pc <= pc;\n'
                             '                end\n'
                             '                default: begin end\n'
                             '            endcase\n'
                             '        end\n'
                             '    end\n'
                             'endmodule\n',
 'verilog/reference/dice_roller.v': '// One roll per sampled low-to-high roll transition; result holds '
                                    'between rolls.\n'
                                    "// The checker validates the public output contract, not this PRNG's "
                                    'exact values.\n'
                                    'module dice_roller (\n'
                                    '    input clk,\n'
                                    '    input rst_n,\n'
                                    '    input [1:0] die_select,\n'
                                    '    input roll,\n'
                                    '    output reg [7:0] rolled_number\n'
                                    ');\n'
                                    '    reg [31:0] random_state;\n'
                                    '    reg roll_previous;\n'
                                    '    reg [7:0] sides;\n'
                                    '    always @* begin\n'
                                    '        case (die_select)\n'
                                    '            0: sides = 4;\n'
                                    '            1: sides = 6;\n'
                                    '            2: sides = 8;\n'
                                    '            default: sides = 20;\n'
                                    '        endcase\n'
                                    '    end\n'
                                    '    always @(posedge clk or negedge rst_n) begin\n'
                                    '        if (!rst_n) begin\n'
                                    "            random_state <= 32'h1ACE_B00C;\n"
                                    '            roll_previous <= 0;\n'
                                    '            rolled_number <= 0;\n'
                                    '        end else begin\n'
                                    '            random_state <= {random_state[30:0],\n'
                                    '                             random_state[31] ^ random_state[21] ^ '
                                    'random_state[1] ^ random_state[0]};\n'
                                    '            roll_previous <= roll;\n'
                                    '            if (roll && !roll_previous)\n'
                                    '                rolled_number <= (random_state % sides) + 1;\n'
                                    '        end\n'
                                    '    end\n'
                                    'endmodule\n',
 'verilog/reference/lfsr.v': 'module lfsr (\n'
                             '    input  clk,\n'
                             '    input  reset_n,\n'
                             '    output reg [7:0] data\n'
                             ');\n'
                             '    always @(posedge clk or negedge reset_n) begin\n'
                             '        if (!reset_n)\n'
                             "            data <= 8'b10001010;\n"
                             '        else\n'
                             '            data <= {data[6:0], data[0] ^ data[3] ^ data[5] ^ data[6]};\n'
                             '    end\n'
                             'endmodule\n',
 'verilog/reference/sequence_generator.v': '// A: shows the first value during reset; each enabled edge '
                                           'advances\n'
                                           'module sequence_generator(input clk, input reset_n, input '
                                           'enable, output reg [7:0] data);\n'
                                           '  reg [2:0] idx;\n'
                                           '  always @(posedge clk or negedge reset_n) begin\n'
                                           "    if (!reset_n) begin idx <= 3'd1; data <= 8'hAF; end\n"
                                           '    else if (enable) begin\n'
                                           '      case (idx)\n'
                                           "        3'd0: data <= 8'hAF; 3'd1: data <= 8'hBC; 3'd2: data <= "
                                           "8'hE2; 3'd3: data <= 8'h78;\n"
                                           "        3'd4: data <= 8'hFF; 3'd5: data <= 8'hE2; 3'd6: data <= "
                                           "8'h0B; 3'd7: data <= 8'h8D;\n"
                                           '      endcase\n'
                                           "      idx <= idx + 3'd1;\n"
                                           '    end\n'
                                           '  end\n'
                                           'endmodule\n',
 'verilog/reference/traffic_light.v': '// Enabled edges count toward each light interval; disabled edges '
                                      'hold phase and timer.\n'
                                      'module traffic_light_fsm (\n'
                                      '    input clk,\n'
                                      '    input reset_n,\n'
                                      '    input enable,\n'
                                      '    output reg red,\n'
                                      '    output reg yellow,\n'
                                      '    output reg green\n'
                                      ');\n'
                                      '    reg [1:0] phase;\n'
                                      '    reg [5:0] elapsed;\n'
                                      '    always @(posedge clk or negedge reset_n) begin\n'
                                      '        if (!reset_n) begin\n'
                                      '            phase <= 0;\n'
                                      '            elapsed <= 0;\n'
                                      '            red <= 1;\n'
                                      '            yellow <= 0;\n'
                                      '            green <= 0;\n'
                                      '        end else if (enable) begin\n'
                                      '            case (phase)\n'
                                      '                0: begin\n'
                                      '                    if (elapsed == 31) begin\n'
                                      '                        phase <= 1;\n'
                                      '                        elapsed <= 0;\n'
                                      '                        red <= 0;\n'
                                      '                        green <= 1;\n'
                                      '                    end else elapsed <= elapsed + 1;\n'
                                      '                end\n'
                                      '                1: begin\n'
                                      '                    if (elapsed == 19) begin\n'
                                      '                        phase <= 2;\n'
                                      '                        elapsed <= 0;\n'
                                      '                        green <= 0;\n'
                                      '                        yellow <= 1;\n'
                                      '                    end else elapsed <= elapsed + 1;\n'
                                      '                end\n'
                                      '                2: begin\n'
                                      '                    if (elapsed == 6) begin\n'
                                      '                        phase <= 0;\n'
                                      '                        elapsed <= 0;\n'
                                      '                        yellow <= 0;\n'
                                      '                        red <= 1;\n'
                                      '                    end else elapsed <= elapsed + 1;\n'
                                      '                end\n'
                                      '                default: begin\n'
                                      '                    phase <= 0;\n'
                                      '                    elapsed <= 0;\n'
                                      '                    red <= 1;\n'
                                      '                    yellow <= 0;\n'
                                      '                    green <= 0;\n'
                                      '                end\n'
                                      '            endcase\n'
                                      '        end\n'
                                      '    end\n'
                                      'endmodule\n',
 'reference/seq/seq_a.v': '// A: shows the first value during reset; each enabled edge advances\n'
                          'module sequence_generator(input clk, input reset_n, input enable, output reg '
                          '[7:0] data);\n'
                          '  reg [2:0] idx;\n'
                          '  always @(posedge clk or negedge reset_n) begin\n'
                          "    if (!reset_n) begin idx <= 3'd1; data <= 8'hAF; end\n"
                          '    else if (enable) begin\n'
                          '      case (idx)\n'
                          "        3'd0: data <= 8'hAF; 3'd1: data <= 8'hBC; 3'd2: data <= 8'hE2; 3'd3: data "
                          "<= 8'h78;\n"
                          "        3'd4: data <= 8'hFF; 3'd5: data <= 8'hE2; 3'd6: data <= 8'h0B; 3'd7: data "
                          "<= 8'h8D;\n"
                          '      endcase\n'
                          "      idx <= idx + 3'd1;\n"
                          '    end\n'
                          '  end\n'
                          'endmodule\n',
 'reference/seq/seq_b.v': '// B: first enabled edge produces the first value; each enabled edge advances\n'
                          'module sequence_generator(input clk, input reset_n, input enable, output reg '
                          '[7:0] data);\n'
                          '  reg [2:0] idx;\n'
                          '  always @(posedge clk or negedge reset_n) begin\n'
                          "    if (!reset_n) begin idx <= 3'd0; data <= 8'h00; end\n"
                          '    else if (enable) begin\n'
                          '      case (idx)\n'
                          "        3'd0: data <= 8'hAF; 3'd1: data <= 8'hBC; 3'd2: data <= 8'hE2; 3'd3: data "
                          "<= 8'h78;\n"
                          "        3'd4: data <= 8'hFF; 3'd5: data <= 8'hE2; 3'd6: data <= 8'h0B; 3'd7: data "
                          "<= 8'h8D;\n"
                          '      endcase\n'
                          "      idx <= idx + 3'd1;\n"
                          '    end\n'
                          '  end\n'
                          'endmodule\n',
 'candidates/bad/inverted_feedback.v': 'module lfsr (\n'
                                       '    input  clk,\n'
                                       '    input  reset_n,\n'
                                       '    output reg [7:0] data\n'
                                       ');\n'
                                       '    always @(posedge clk or negedge reset_n) begin\n'
                                       '        if (!reset_n)\n'
                                       "            data <= 8'b10001010;\n"
                                       '        else\n'
                                       '            data <= {data[6:0], ~(data[0] ^ data[3] ^ data[5] ^ '
                                       'data[6])};\n'
                                       '    end\n'
                                       'endmodule\n',
 'candidates/bad/lfsr_badtap.v': 'module lfsr (\n'
                                 '    input  clk,\n'
                                 '    input  reset_n,\n'
                                 '    output reg [7:0] data\n'
                                 ');\n'
                                 '    always @(posedge clk or negedge reset_n) begin\n'
                                 '        if (!reset_n)\n'
                                 "            data <= 8'b10001010;\n"
                                 '        else\n'
                                 '            data <= {data[6:0], data[0] ^ data[3] ^ data[4] ^ data[6]};\n'
                                 '    end\n'
                                 'endmodule\n',
 'candidates/bad/lfsr_initial_hack.v': 'module lfsr (\n'
                                       '    input  clk,\n'
                                       '    input  reset_n,\n'
                                       '    output reg [7:0] data\n'
                                       ');\n'
                                       "    initial data = 8'b10001010;\n"
                                       '\n'
                                       '    always @(posedge clk) begin\n'
                                       '        if (!reset_n)\n'
                                       "            data <= 8'b10001010;\n"
                                       '        else\n'
                                       '            data <= {data[6:0], data[0] ^ data[3] ^ data[5] ^ '
                                       'data[6]};\n'
                                       '    end\n'
                                       'endmodule\n',
 'candidates/bad/lfsr_syncreset.v': 'module lfsr (\n'
                                    '    input  clk,\n'
                                    '    input  reset_n,\n'
                                    '    output reg [7:0] data\n'
                                    ');\n'
                                    '    always @(posedge clk) begin\n'
                                    '        if (!reset_n)\n'
                                    "            data <= 8'b10001010;\n"
                                    '        else\n'
                                    '            data <= {data[6:0], data[0] ^ data[3] ^ data[5] ^ '
                                    'data[6]};\n'
                                    '    end\n'
                                    'endmodule\n',
 'candidates/bad/lfsr_two_drivers.v': 'module lfsr (\n'
                                      '    input clk,\n'
                                      '    input reset_n,\n'
                                      '    output reg [7:0] data\n'
                                      ');\n'
                                      '    always @(posedge clk) begin\n'
                                      '        data <= {data[6:0], data[0] ^ data[3] ^ data[5] ^ data[6]};\n'
                                      '    end\n'
                                      '\n'
                                      '    always @(reset_n) begin\n'
                                      "        data <= 8'b10001010;\n"
                                      '    end\n'
                                      'endmodule\n',
 'candidates/bad/sequence_generator/bad_item.v': '// A: shows the first value during reset; each enabled '
                                                 'edge advances\n'
                                                 'module sequence_generator(input clk, input reset_n, input '
                                                 'enable, output reg [7:0] data);\n'
                                                 '  reg [2:0] idx;\n'
                                                 '  always @(posedge clk or negedge reset_n) begin\n'
                                                 "    if (!reset_n) begin idx <= 3'd1; data <= 8'hAF; end\n"
                                                 '    else if (enable) begin\n'
                                                 '      case (idx)\n'
                                                 "        3'd0: data <= 8'hAF; 3'd1: data <= 8'hBC; 3'd2: "
                                                 "data <= 8'hE2; 3'd3: data <= 8'h78;\n"
                                                 "        3'd4: data <= 8'hFF; 3'd5: data <= 8'hE3; 3'd6: "
                                                 "data <= 8'h0B; 3'd7: data <= 8'h8D;\n"
                                                 '      endcase\n'
                                                 "      idx <= idx + 3'd1;\n"
                                                 '    end\n'
                                                 '  end\n'
                                                 'endmodule\n',
 'candidates/bad/sequence_generator/ignores_enable.v': '// A: shows the first value during reset; each '
                                                       'enabled edge advances\n'
                                                       'module sequence_generator(input clk, input reset_n, '
                                                       'input enable, output reg [7:0] data);\n'
                                                       '  reg [2:0] idx;\n'
                                                       '  always @(posedge clk or negedge reset_n) begin\n'
                                                       "    if (!reset_n) begin idx <= 3'd1; data <= 8'hAF; "
                                                       'end\n'
                                                       '    else begin\n'
                                                       '      case (idx)\n'
                                                       "        3'd0: data <= 8'hAF; 3'd1: data <= 8'hBC; "
                                                       "3'd2: data <= 8'hE2; 3'd3: data <= 8'h78;\n"
                                                       "        3'd4: data <= 8'hFF; 3'd5: data <= 8'hE2; "
                                                       "3'd6: data <= 8'h0B; 3'd7: data <= 8'h8D;\n"
                                                       '      endcase\n'
                                                       "      idx <= idx + 3'd1;\n"
                                                       '    end\n'
                                                       '  end\n'
                                                       'endmodule\n',
 'candidates/bad/sequence_generator/no_wrap.v': '// A: shows the first value during reset; each enabled edge '
                                                'advances\n'
                                                'module sequence_generator(input clk, input reset_n, input '
                                                'enable, output reg [7:0] data);\n'
                                                '  reg [2:0] idx;\n'
                                                '  always @(posedge clk or negedge reset_n) begin\n'
                                                "    if (!reset_n) begin idx <= 3'd1; data <= 8'hAF; end\n"
                                                '    else if (enable) begin\n'
                                                '      case (idx)\n'
                                                "        3'd0: data <= 8'hAF; 3'd1: data <= 8'hBC; 3'd2: "
                                                "data <= 8'hE2; 3'd3: data <= 8'h78;\n"
                                                "        3'd4: data <= 8'hFF; 3'd5: data <= 8'hE2; 3'd6: "
                                                "data <= 8'h0B; 3'd7: data <= 8'h8D;\n"
                                                '      endcase\n'
                                                "      if (idx != 7) idx <= idx + 3'd1;\n"
                                                '    end\n'
                                                '  end\n'
                                                'endmodule\n',
 'candidates/bad/sequence_generator/reset_position.v': '// A: shows the first value during reset; each '
                                                       'enabled edge advances\n'
                                                       'module sequence_generator(input clk, input reset_n, '
                                                       'input enable, output reg [7:0] data);\n'
                                                       '  reg [2:0] idx;\n'
                                                       '  always @(posedge clk or negedge reset_n) begin\n'
                                                       "    if (!reset_n) begin idx <= 3'd0; data <= 8'hAF; "
                                                       'end\n'
                                                       '    else if (enable) begin\n'
                                                       '      case (idx)\n'
                                                       "        3'd0: data <= 8'hAF; 3'd1: data <= 8'hBC; "
                                                       "3'd2: data <= 8'hE2; 3'd3: data <= 8'h78;\n"
                                                       "        3'd4: data <= 8'hFF; 3'd5: data <= 8'hE2; "
                                                       "3'd6: data <= 8'h0B; 3'd7: data <= 8'h8D;\n"
                                                       '      endcase\n'
                                                       "      idx <= idx + 3'd1;\n"
                                                       '    end\n'
                                                       '  end\n'
                                                       'endmodule\n',
 'candidates/bad/sequence_generator/restart_on_pause.v': '// A: shows the first value during reset; each '
                                                         'enabled edge advances\n'
                                                         'module sequence_generator(input clk, input '
                                                         'reset_n, input enable, output reg [7:0] data);\n'
                                                         '  reg [2:0] idx;\n'
                                                         '  always @(posedge clk or negedge reset_n) begin\n'
                                                         "    if (!reset_n) begin idx <= 3'd1; data <= "
                                                         "8'hAF; end\n"
                                                         '    else if (enable) begin\n'
                                                         '      case (idx)\n'
                                                         "        3'd0: data <= 8'hAF; 3'd1: data <= 8'hBC; "
                                                         "3'd2: data <= 8'hE2; 3'd3: data <= 8'h78;\n"
                                                         "        3'd4: data <= 8'hFF; 3'd5: data <= 8'hE2; "
                                                         "3'd6: data <= 8'h0B; 3'd7: data <= 8'h8D;\n"
                                                         '      endcase\n'
                                                         "      idx <= idx + 3'd1;\n"
                                                         "    end else idx <= 3'd1;\n"
                                                         '  end\n'
                                                         'endmodule\n',
 'candidates/bad/shift_wrong_direction.v': 'module lfsr (\n'
                                           '    input  clk,\n'
                                           '    input  reset_n,\n'
                                           '    output reg [7:0] data\n'
                                           ');\n'
                                           '    always @(posedge clk or negedge reset_n) begin\n'
                                           '        if (!reset_n)\n'
                                           "            data <= 8'b10001010;\n"
                                           '        else\n'
                                           '            data <= {data[0] ^ data[3] ^ data[5] ^ data[6], '
                                           'data[7:1]};\n'
                                           '    end\n'
                                           'endmodule\n',
 'candidates/lfsr.v': 'module lfsr (\n'
                      '    input  clk,\n'
                      '    input  reset_n,\n'
                      '    output reg [7:0] data\n'
                      ');\n'
                      '    always @(posedge clk or negedge reset_n) begin\n'
                      '        if (!reset_n)\n'
                      "            data <= 8'b10001010;\n"
                      '        else\n'
                      '            data <= {data[6:0], data[0] ^ data[3] ^ data[5] ^ data[6]};\n'
                      '    end\n'
                      'endmodule\n',
 'baseline_lfsr.txt': 'candidates/lfsr.v: 1.0\n'
                      '    correct\n'
                      'candidates/bad/inverted_feedback.v: 0.4\n'
                      '    reset_midrun: requirement: reset during a run must restore the initial state '
                      'immediately, hold it across a clock edge, and restart the specified sequence after '
                      'release. Observed: reset restored and held the initial state correctly; the sequence '
                      'after restart fails at restart cycle 1 after reset release, data is wrong in bit(s) '
                      '[0]; the other 7 bits are correct.\n'
                      '    first_step: requirement: on each clock edge after reset_n is released, data must '
                      'advance by one step of the LFSR in the spec. Observed: on the first edge after '
                      'release, data is wrong in bit(s) [0]; the other 7 bits are correct.\n'
                      '    sequence: requirement: every clock edge must advance data by one step of the LFSR '
                      'in the spec. Observed: at cycle 1 after release, data is wrong in bit(s) [0]; the '
                      'other 7 bits are correct.\n'
                      '    Facts and score verified\n'
                      'candidates/bad/lfsr_badtap.v: 0.55\n'
                      '    reset_midrun: requirement: reset during a run must restore the initial state '
                      'immediately, hold it across a clock edge, and restart the specified sequence after '
                      'release. Observed: reset restored and held the initial state correctly; the sequence '
                      'after restart fails at restart cycle 2 after reset release, data is wrong in bit(s) '
                      '[0]; the other 7 bits are correct.\n'
                      '    sequence: requirement: every clock edge must advance data by one step of the LFSR '
                      'in the spec. Observed: at cycle 2 after release, data is wrong in bit(s) [0]; the '
                      'other 7 bits are correct.\n'
                      '    Facts and score verified\n'
                      'candidates/bad/lfsr_initial_hack.v: 0.8\n'
                      '    reset_midrun: requirement: reset during a run must restore the initial state '
                      'immediately, hold it across a clock edge, and restart the specified sequence after '
                      'release. Observed: at 2573 ns, one ns after reset assertion with clk low and no clock '
                      'edge, data is wrong in bit(s) [7, 5, 0]; the other 5 bits are correct.\n'
                      '    In your design, data is assigned in 2 place(s): line 6, an initial block; line 8, '
                      'a block triggered by posedge clk. (data is driven from more than one block) (data '
                      'uses a blocking assignment on line 6)\n'
                      '    Facts and score verified\n'
                      'candidates/bad/lfsr_syncreset.v: 0.65\n'
                      '    reset_immediate: requirement: while reset_n is 0, data must equal the initial '
                      'state at every moment, including before the first clock edge. Observed: at 2 ns, with '
                      'reset_n at 0 and no clock edge yet, data is undefined (X).\n'
                      '    In your design, data is assigned in 1 place(s): line 6, a block triggered by '
                      'posedge clk.\n'
                      '    reset_midrun: requirement: reset during a run must restore the initial state '
                      'immediately, hold it across a clock edge, and restart the specified sequence after '
                      'release. Observed: at 2573 ns, one ns after reset assertion with clk low and no clock '
                      'edge, data is wrong in bit(s) [7, 5, 0]; the other 5 bits are correct.\n'
                      '    In your design, data is assigned in 1 place(s): line 6, a block triggered by '
                      'posedge clk.\n'
                      '    Facts and score verified\n'
                      'candidates/bad/lfsr_two_drivers.v: 0.65\n'
                      '    reset_value: requirement: while reset_n is 0, data must equal the initial state '
                      'from the spec. Observed: with reset_n held at 0, after a clock edge, data is wrong in '
                      'bit(s) [7, 4, 3, 2, 1, 0]; the other 2 bits are correct.\n'
                      '    In your design, data is assigned in 2 place(s): line 6, a block triggered by '
                      'posedge clk; line 10, a block triggered by reset_n. (data is driven from more than '
                      'one block)\n'
                      '    reset_midrun: requirement: reset during a run must restore the initial state '
                      'immediately, hold it across a clock edge, and restart the specified sequence after '
                      'release. Observed: at 2577 ns, after a clock edge with reset held, data is wrong in '
                      'bit(s) [7, 4, 3, 2, 1, 0]; the other 2 bits are correct.\n'
                      '    In your design, data is assigned in 2 place(s): line 6, a block triggered by '
                      'posedge clk; line 10, a block triggered by reset_n. (data is driven from more than '
                      'one block)\n'
                      '    Facts and score verified\n'
                      'candidates/bad/shift_wrong_direction.v: 0.4\n'
                      '    reset_midrun: requirement: reset during a run must restore the initial state '
                      'immediately, hold it across a clock edge, and restart the specified sequence after '
                      'release. Observed: reset restored and held the initial state correctly; the sequence '
                      'after restart fails at restart cycle 1 after reset release, data is wrong in bit(s) '
                      '[7, 6, 4]; the other 5 bits are correct.\n'
                      '    first_step: requirement: on each clock edge after reset_n is released, data must '
                      'advance by one step of the LFSR in the spec. Observed: on the first edge after '
                      'release, data is wrong in bit(s) [7, 6, 4]; the other 5 bits are correct.\n'
                      '    sequence: requirement: every clock edge must advance data by one step of the LFSR '
                      'in the spec. Observed: at cycle 1 after release, data is wrong in bit(s) [7, 6, 4]; '
                      'the other 5 bits are correct.\n'
                      '    Facts and score verified\n'
                      '\n'
                      'SELFTEST PASSED\n',
 'README.md': '# Single-file Verilog checker and agent\n'
              '\n'
              'Only checker.py is needed to start a fresh copy. It contains the checker, agent,\n'
              'taxonomy lookup, feedback construction, embedded selftests, all JSON\n'
              'defaults, and the existing specs/reference designs/testbenches. Python 3.11+\n'
              'and Icarus Verilog (iverilog and vvp) must be installed separately.\n'
              '\n'
              'On first execution it creates missing resources beside itself. JSON defaults\n'
              'are Python variables named KNOWLEDGE, TAXONOMY, PRINCIPLES,\n'
              'EXPERIMENT_MANIFEST, and REFERENCE_METADATA. Existing runtime JSON files are\n'
              'kept, and knowledge/taxonomy are read again for each repair prompt. Edit those\n'
              'runtime files to change retrieval dynamically. Only selected rules enter the\n'
              'model prompt. Reference source stays on the grading side.\n'
              '\n'
              'Initialize files without calling a model:\n'
              '\n'
              '    python checker.py --init\n'
              '\n'
              'Generate one candidate and check it (local model must already be serving):\n'
              '\n'
              '    python checker.py --agent --problem lfsr --repeat 1 --rounds 1 --explain --principles\n'
              '\n'
              'Run the measured LFSR setting or try the CPU:\n'
              '\n'
              '    python checker.py --agent --problem lfsr --repeat 5 --rounds 5 --explain --principles\n'
              '    python checker.py --agent --problem cpu8 --repeat 1 --rounds 20 --explain --principles\n'
              '\n'
              'Other prepared agent tasks: sequence_generator, traffic_light, dice_roller.\n'
              'Use --base-url URL and --model NAME to select another model endpoint.\n'
              '\n'
              'Grade an existing candidate or use an arbitrary spec:\n'
              '\n'
              '    python checker.py --spec prompts/specs/LFSR.md --dut candidate.v\n'
              '    python checker.py --agent --spec design.md --tb design_tb.v --ref reference.v --repeat 1 '
              '--rounds 5\n'
              '    python checker.py --spec design.md --tb design_tb.v --ref reference.v --dut candidate.v\n'
              '\n'
              'Run just a supplied testbench, with no reference comparison:\n'
              '\n'
              '    python checker.py --run-tb --tb TestBench/cpu8_tb.v --dut candidate.v\n'
              '\n'
              'Batch and offline selftests:\n'
              '\n'
              '    python checker.py --run-all --mode checker\n'
              '    python checker.py --run-all --mode agent --repeat 1 --rounds 1\n'
              '    python checker.py\n'
              '    python checker.py --selftest\n'
              '\n'
              'The agent feedback and dynamic --principles behavior are unchanged. Expected\n'
              'reference values and reference source are not included in repair prompts.\n'
              '\n'
              'Plain output: log/PROBLEM.log (replaced by the next run of that problem).\n'
              'Complete attempt history: log/runs/<unique-run>/events.jsonl.\n'
              'Exact model-facing messages: prompts/runs/<unique-run>/.\n'
              'Generated designs: candidates/gen/<unique-run>/.\n'
              'The existing logs and candidates have been copied without rewriting records.\n'
              'Historical paths/hashes in those records describe the original experiment.\n',
 'writeup.md': '**Verilog generation with checker feedback — recorded results, 10 October 2026**\n'
               '\n'
               'The best recorded LFSR setting, `--explain --principles`, solved **5/5 runs**. An 8-bit CPU '
               'experiment with the same feedback setting solved **0/1 runs after 20 attempts**, ending at '
               '**0.75**. These are saved experiment results; no new model calls were made for this report.\n'
               '\n'
               'Both experiments used `Qwen/Qwen3-8B` at `http://localhost:8000/v1`, temperature 0.7, '
               'thinking disabled, a 1,500-token output limit, and an 8,000-token configured context. '
               'Checking uses Icarus Verilog on the CPU. Each run starts with empty history.\n'
               '\n'
               '**How the loop works**\n'
               '\n'
               '1. `checker.py` builds a prompt from the Markdown specification, the module interface, and '
               'any public clarification stored in the adjacent reference JSON. `agent.py` sends these '
               'messages to Qwen. The initial LFSR prompt explicitly states immediate reset, the reset '
               'value, and the recurrence.\n'
               '2. Qwen returns Verilog. The checker extracts the module and the agent saves it under '
               '`candidates/gen/`. The checker compiles a generated testbench containing both the candidate '
               'and an independent reference design. It drives identical stimulus, changes inputs while the '
               'clock is low, and samples outputs 1 ns after rising edges. Separate observations test '
               'immediate reset and reset during execution. These runs used seed 0, 256 primary edges, and '
               '128 additional cycles in the random segment.\n'
               '3. The checker compares candidate outputs against reference outputs and returns the first '
               'mismatch for each failed requirement, including undefined outputs, differing bits, timing '
               "context, and applicable facts about the candidate's own code. A score is the sum of weights "
               'for passing check categories, rather than the proportion of cycles that pass. A score of 1.0 '
               'also requires the supplied official testbench to pass.\n'
               '4. Before the next attempt, the checker reloads `taxonomy.json` and `knowledge.json`, '
               'matches current failure signatures and applicable categories, and selects at most two '
               'relevant principles. It sends the specification, latest candidate, feedback, and selected '
               'principles; `--explain` asks for a diagnosis before the revised module. The full knowledge '
               'table is never inserted into the prompt. The loop ends at 1.0 or the attempt limit.\n'
               '\n'
               'The reference `.v` supplies the grading comparison; its source and sampled expected outputs '
               'are not sent to Qwen in these recorded runs. Repair guidance comes from observed failures, '
               'candidate code facts, and retrieved principles. Public specification values remain in the '
               'prompt.\n'
               '\n'
               '**Best LFSR result**\n'
               '\n'
               'Five runs allowed up to five attempts each. All initial candidates scored 0.70 and used '
               'reset logic that waited for a clock edge. The final candidates handled reset immediately and '
               'passed the generic checks and official testbench. Four runs needed one repair; the fifth '
               'repeated its initial candidate once and needed two repairs.\n'
               '\n'
               '| Run (log index) | Scores by attempt | Attempts | Official testbench on final candidate |\n'
               '|---|---|---:|---|\n'
               '| 0 | 0.70 → 1.00 | 2 | Pass |\n'
               '| 1 | 0.70 → 1.00 | 2 | Pass |\n'
               '| 2 | 0.70 → 1.00 | 2 | Pass |\n'
               '| 3 | 0.70 → 1.00 | 2 | Pass |\n'
               '| 4 | 0.70 → 0.70 → 1.00 | 3 | Pass |\n'
               '\n'
               'Total: **11 attempts**, averaging **2.2 attempts per solve**. Final scores were all 1.00. '
               'This reports the best recorded condition only; it does not isolate the effect of principles '
               'from `--explain` or establish performance on unseen tasks. LFSR-specific knowledge keys were '
               'enabled.\n'
               '\n'
               'The selected rules were:\n'
               '\n'
               '> Principle: An output required before the first clock edge must become defined '
               'independently of a clocked transition.\n'
               '>\n'
               '> Principle: A reset specified to act independently of the clock must restore the required '
               'state between clock edges.\n'
               '\n'
               'Evidence: [LFSR console log](log/lfsr.log) and [complete prompts, replies, candidates, '
               'settings, and '
               'scores](log/runs/md_generic_v4_dynamic_taxonomy_lfsr_20261010T200026Z_1791662426884934708/events.jsonl).\n'
               '\n'
               '**Failed 8-bit CPU experiment**\n'
               '\n'
               'The [CPU specification](prompts/specs/cpu8.md) describes an accumulator core with externally '
               'supplied instructions, arithmetic, jumps, output, halt, enable, and reset. This was one run '
               'with 20 attempts, rather than a five-run success-rate measurement.\n'
               '\n'
               '| Attempts (zero-based rounds) | Score | Recorded outcome |\n'
               '|---|---:|---|\n'
               '| 0 | 0.00 | Compilation failed: instruction array/interface errors. |\n'
               '| 1–2 | 0.20 | Compiled, but failed reset and execution checks; multiple output drivers were '
               'reported. |\n'
               '| 3–14 | 0.00 | Repeated compilation syntax errors. |\n'
               '| 15 | 0.00 | No complete module returned. |\n'
               '| 16 | 0.00 | Instruction array/interface compilation errors returned. |\n'
               '| 17–19 | 0.75 | Compiled, but failed random-input and input-toggle checks. |\n'
               '\n'
               'There were **14 compilation failures**, one missing-module response, and five compiled '
               'attempts with functional failures. No candidate passed the official testbench. The final '
               'feedback reports `pc` undefined at cycle 259 and `acc` undefined at cycle 387, two cycles '
               'after enable went low. The final attempt repeated the preceding candidate. Thus **0.75 is a '
               'failed result**, not a solved CPU.\n'
               '\n'
               'Evidence: [CPU console log](log/cpu8.log), [complete attempt '
               'records](log/runs/md_generic_v4_dynamic_taxonomy_cpu8_20261010T201926Z_1791663566973007868/events.jsonl), '
               'and [final '
               'candidate](candidates/gen/md_generic_v4_dynamic_taxonomy_cpu8_20261010T201926Z_1791663566973007868/cpu8/cpu8_0_19.v).\n'
               '\n'
               '**Commands for the installed project**\n'
               '\n'
               'With the local model already serving, reproduce the experiment settings:\n'
               '\n'
               '```bash\n'
               'cd /workspace/final\n'
               'python checker.py --agent --problem lfsr --repeat 5 --rounds 5 --explain --principles\n'
               'python checker.py --agent --problem cpu8 --repeat 1 --rounds 20 --explain --principles\n'
               '```\n'
               '\n'
               'Run the other prepared problems individually:\n'
               '\n'
               '```bash\n'
               'cd /workspace/final\n'
               'python checker.py --agent --problem sequence_generator --repeat 5 --rounds 5 --explain '
               '--principles\n'
               'python checker.py --agent --problem traffic_light --repeat 5 --rounds 5 --explain '
               '--principles\n'
               'python checker.py --agent --problem dice_roller --repeat 5 --rounds 5 --explain '
               '--principles\n'
               '```\n'
               '\n'
               'Use `--repeat 1 --rounds 1` for one prompt followed by one check. For a run that survives '
               'disconnection:\n'
               '\n'
               '```bash\n'
               'cd /workspace/final\n'
               'nohup python checker.py --agent --problem sequence_generator --repeat 5 --rounds 5 --explain '
               '--principles > log/sequence_generator.launch.log 2>&1 < /dev/null &\n'
               'tail -f log/sequence_generator.log\n'
               '```\n'
               '\n'
               'Check the four prepared references without calling Qwen:\n'
               '\n'
               '```bash\n'
               'cd /workspace/final\n'
               'python checker.py --run-all --mode checker\n'
               '```\n'
               '\n'
               'Run an existing official testbench on your own saved design without a reference comparison:\n'
               '\n'
               '```bash\n'
               'cd /workspace/final\n'
               'python checker.py --run-tb --tb TestBench/cpu8_tb.v --dut /path/to/your_cpu8.v\n'
               '```\n'
               '\n'
               "View results with `tail -f log/PROBLEM.log`. Each new run replaces that problem's short log; "
               'complete attempt records remain under uniquely named `log/runs/` directories. Saved '
               'model-facing prompts are under `prompts/`. The checker has been updated since the recorded '
               'experiments, so the commands above start new runs with the installed version. Integration of '
               'the newly merged TB branch remains stopped.\n'}


def initialize_files():
    """Write defaults only for absent files; preserve live JSON edits."""
    defaults = dict(REFERENCE_METADATA)
    defaults.update({
        'knowledge.json': KNOWLEDGE,
        'taxonomy.json': TAXONOMY,
        'principles.json': PRINCIPLES,
        'experiments/md_generic_v4_dynamic_taxonomy.json': EXPERIMENT_MANIFEST,
    })
    created = []
    for name, data in defaults.items():
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open('x', encoding='utf-8') as stream:
                json.dump(data, stream, indent=2)
                stream.write('\n')
        except FileExistsError:
            if not path.is_file():
                raise
        else:
            created.append(name)
    for name, text in BUNDLED_TEXT_FILES.items():
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open('x', encoding='utf-8') as stream:
                stream.write(text)
        except FileExistsError:
            if not path.is_file():
                raise
        else:
            created.append(name)
    (ROOT / 'log').mkdir(exist_ok=True)
    return created


initialize_files()


def _namespace(name, members):
    """Internal API namespaces; no sibling Python files are imported."""
    module = types.ModuleType(name)
    module.__dict__.update(members)
    module.__file__ = __file__
    sys.modules[name] = module
    return module


# ---- taxonomy implementation ----
"""Dynamic category classification, signature descriptions, and bounded rule retrieval."""
import hashlib
import json
import pathlib
import re

def _taxonomy_read(path):
    return _taxonomy_snapshot(path)[0]

def _taxonomy_snapshot(path):
    raw = pathlib.Path(path).read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError('taxonomy and knowledge files must be JSON objects')
    structured = 'categories' in data or 'signatures' in data
    if structured:
        if not isinstance(data.get('categories', []), list) or not isinstance(data.get('signatures', {}), dict):
            raise ValueError('taxonomy needs a category list and a signature dictionary')
    elif any((not isinstance(key, str) or ':' not in key or (not isinstance(rule, str)) for key, rule in data.items())):
        raise ValueError('knowledge must map category:signature to a rule sentence')
    return (data, hashlib.sha256(raw).hexdigest())

def _taxonomy_categories(text, ports=(), path='taxonomy.json', overrides=(), *, data=None, exclude=()):
    """Category keywords, port hints, and priority come from the live taxonomy JSON."""
    data = _taxonomy_read(path) if data is None else data
    text = text.lower().replace('_', ' ')
    names = [name.lower() for name, _, _ in ports]
    found = list(overrides)
    if 'categories' in data:
        for category in data['categories']:
            keywords = category.get('spec_keywords', [])
            hints = category.get('port_hints', [])
            matched = any((re.search('\\b' + re.escape(keyword) + '\\b', text, re.I) for keyword in keywords))
            matched |= any((hint.lower() == name or hint.lower() in name.split('_') for hint in hints for name in names))
            if category['name'] == 'combinational' and category.get('rule'):
                matched |= not any((re.search('clk|clock', name) for name, direction, _ in ports if direction == 'input'))
            if matched:
                found.append(category['name'])
    else:
        available = dict.fromkeys((key.split(':', 1)[0] for key in data))
        for category in available:
            words = category.replace('_', ' ').split()
            if category not in ('*', 'generic') and all((re.search('\\b' + re.escape(word) + '\\w*', text) for word in words)):
                found.append(category)
    return [name for name in dict.fromkeys(found + ['generic']) if name not in exclude]

def _taxonomy_select(path, applicable, signatures, *, limit=2, max_bytes=1200, data=None, knowledge=None, show_detect_text=False):
    """Select current fired signatures only; never send the entire JSON to a model."""
    data = _taxonomy_read(path) if data is None else data
    definitions = data.get('signatures', {})
    if 'categories' in data or 'signatures' in data:
        rules = data.get('knowledge', data.get('rules', knowledge or {}))
        lookup = data.get('lookup', {})
        cap = lookup.get('max_rules')
        if cap is None:
            match = re.search('at most\\s+(\\d+)', lookup.get('rule', ''), re.I)
            cap = int(match[1]) if match else limit
        limit = min(limit, cap)
    else:
        rules = data
    if limit <= 0:
        return []
    selected, used, budget = ([], set(), 0)
    for signature in dict.fromkeys(signatures):
        keys = [f'{category}:{signature}' for category in applicable] + [f'*:{signature}']
        key = next((key for key in keys if isinstance(rules.get(key), str) and rules[key].strip()), None)
        rule = rules[key] if key else None
        definition = definitions.get(signature, {})
        description = definition.get('detect', '')
        if rule in used and rule is not None:
            continue
        if not rule:
            continue
        entry = dict(key=key, signature=signature, rule=rule, description=description, source=definition.get('source'))
        size = len(_taxonomy_entry_text(entry, show_detect_text=show_detect_text).encode()) + 1
        if budget + size > max_bytes:
            continue
        selected.append(entry)
        if rule:
            used.add(rule)
        budget += size
        if len(selected) >= limit:
            break
    return selected

def _taxonomy_entry_text(entry, *, show_detect_text=False):
    if not entry.get('rule'):
        return ''
    lines = []
    if show_detect_text and entry.get('description'):
        lines.append(f"Detected signature {entry['signature']}: {entry['description']}.")
    if entry.get('rule'):
        lines.append('Principle: ' + entry['rule'])
    return '\n'.join(lines)

def _taxonomy_knowledge_report(current='knowledge.json', original='knowledge_original.json'):
    """Compare every key without modifying either input file."""
    current, original = (pathlib.Path(current), pathlib.Path(original))
    report = dict(current=str(current), original=str(original), current_md5=hashlib.md5(current.read_bytes()).hexdigest(), original_md5=None, original_expected_md5_prefix='a38ea61b', added={}, removed={}, changed={})
    if not original.is_file():
        report['status'] = 'original_missing'
        return report
    report['original_md5'] = hashlib.md5(original.read_bytes()).hexdigest()
    report['original_matches_expected_prefix'] = report['original_md5'].startswith('a38ea61b')
    before, after = (_taxonomy_read(original), _taxonomy_read(current))
    report['added'] = {key: after[key] for key in sorted(after.keys() - before.keys())}
    report['removed'] = {key: before[key] for key in sorted(before.keys() - after.keys())}
    report['changed'] = {key: dict(before=before[key], after=after[key]) for key in sorted(before.keys() & after.keys()) if before[key] != after[key]}
    report['status'] = 'compared'
    return report

taxonomy = _namespace('taxonomy', {'read': _taxonomy_read, 'snapshot': _taxonomy_snapshot, 'categories': _taxonomy_categories, 'select': _taxonomy_select, 'entry_text': _taxonomy_entry_text, 'knowledge_report': _taxonomy_knowledge_report})

# ---- checker engine and prompt construction ----
"""
checker.py - grades a candidate Verilog design and explains what is wrong.

How it grades: it generates its own probe testbench, drives the design, and compares
every output against a Python model or a Verilog reference driven by the same inputs.
A 1.0 also requires the official testbench to pass when one is available.

What the feedback says: for each failed check, the REQUIREMENT that check enforces and
what was OBSERVED (signal, time or cycle, which bits are wrong). It never prints the
expected value and never suggests code. The reference design and reference model are
used only to grade and to self-test the checker; the model never sees them.

Problem definitions supply ports, reset behavior, reference models, stimulus
segments, feedback requirements, and weights. The probe and grader interpret this
data without dispatching to a problem-specific checker.
"""
import copy, json, random, re, subprocess, pathlib, sys, time, hashlib, contextlib, shutil
from datetime import datetime, timezone
ROOT = pathlib.Path(__file__).resolve().parent
CODE_ONLY = '\nReturn only the complete Verilog module in one code block.'
EXTRA_HINT = ' Reset is asynchronous and takes effect without a clock edge.'
LEGACY_SPECS = {'mine': "Write a Verilog module named `lfsr` for an 8-bit linear feedback shift register.\nPorts: input clk, input reset_n (active low), output reg [7:0] data.\nOn reset_n low, data is set to 8'b10001010.{extra}\nOn each rising clock edge with reset_n high, shift data left by one and set the new\nbit 0 to the XOR of the old bits 0, 3, 5 and 6.", 'bench': 'I am trying to create a Verilog model for an LFSR. It must meet the following specifications:\n- Inputs: Clock, Active-low reset\n- Outputs: Data (8-bits)\nThe initial state should be 10001010, and the taps should be at locations 1, 4, 6, and 7.{extra}\nUse module name `lfsr` with ports: input clk, input reset_n, output reg [7:0] data.'}
LEGACY_CODE_ONLY = '\nReturn only the Verilog module in one code block.'
LEGACY_V1_SPEC = "Write a Verilog module named `lfsr` for an 8-bit linear feedback shift register.\nPorts: input clk, input reset_n (active low), output reg [7:0] data.\nOn reset_n low, data is set to 8'b10001010.{extra}\nOn each rising clock edge with reset_n high, shift data left by one and set the new\nbit 0 to the XOR of the old bits 0, 3, 5 and 6.\nReturn only the Verilog module in one code block."

def extract(text):
    blocks = re.findall('```(?:verilog|systemverilog|v)?\\s*\\n(.*?)```', text, re.S)
    blocks = [block for block in blocks if re.search('\\bmodule\\b', block)]
    if blocks:
        return blocks[-1]
    return text if re.search('\\bmodule\\b', text) and 'endmodule' in text else None

def _bytes(text):
    return len(text.encode('utf-8'))

def _fit(text, budget):
    return text.encode('utf-8')[:max(0, budget)].decode('utf-8', errors='ignore')

def _bounded_prompt(spec, history, show_code=True, explain=False, *, rules=(), max_input_bytes=None, show_detect_text=False):
    """Only the latest attempt enters context; full history remains in the log.

    UTF-8 bytes conservatively bound byte-token input length. Reserve generation
    tokens and chat overhead outside this budget; never truncate the specification.
    """
    if not history:
        content = spec + CODE_ONLY
    else:
        code, score, feedback = history[-1]
        instruction = 'First, for each failure, say in one sentence which part of the design causes it. Then end with the complete corrected Verilog module in one code block.' if explain else 'Fix the failures and return the complete corrected Verilog module in one code block.'
        content = spec + '\n\n' + instruction
        rule_text = '\n'.join((taxonomy.entry_text(entry, show_detect_text=show_detect_text) for entry in rules))
        if rule_text:
            content += '\n\nRules selected for the current failures:\n' + rule_text
        header = f'\n\nLatest checker result (score {score:.2f}):\n'
        available = max_input_bytes - _bytes(content + header) if max_input_bytes is not None else 4000
        if available < 160:
            raise ValueError('specification and selected rules exceed the prompt budget; increase --context-tokens or shorten the specification')
        feedback_budget = min(3000, available)
        visible = _fit(feedback, feedback_budget)
        if visible != feedback:
            visible = _fit(visible, feedback_budget - 60) + '\n[More feedback is recorded in the log.]'
        content += header + visible
        if show_code and code:
            header = '\n\nPrevious attempt:\n```verilog\n'
            end = '\n```'
            remaining = max_input_bytes - _bytes(content + header + end) if max_input_bytes is not None else _bytes(code)
            if remaining > 200:
                preview = code if _bytes(code) <= remaining else _fit(code, remaining - 80) + '\n// Excerpt: the full previous attempt is saved in the log.'
                content += header + preview + end
        if len(history) > 1 and history[-1][0].strip() == history[-2][0].strip():
            note = '\nThe last two attempts were identical; revise the logic.'
            if max_input_bytes is None or _bytes(content + note) <= max_input_bytes:
                content += note
    if max_input_bytes is not None and _bytes(content) > max_input_bytes:
        raise ValueError('the Markdown specification exceeds the input budget; increase --context-tokens or shorten it')
    return [{'role': 'user', 'content': content}]

def _legacy_prompt(spec, history, show_code=True, explain=False):
    if not history:
        return [{'role': 'user', 'content': spec + LEGACY_CODE_ONLY}]
    code, score, fb = history[-1]
    msg = spec
    if show_code:
        msg += f'\n\nYour previous attempt:\n```verilog\n{code}\n```'
    msg += f'\n\nA checker tested your previous attempt (score {score:.2f}):\n{fb}\n'
    if len(history) > 1:
        msg += '\nEarlier feedback:\n' + '\n'.join((f'- {h[2].splitlines()[0]}' for h in history[:-1]))
    if len(history) > 1 and history[-1][0].strip() == history[-2][0].strip():
        msg += '\nYour last two attempts were identical and both failed. Change the logic, not just the formatting.'
    if explain:
        msg += '\nFirst, for each failure above, say in one sentence which part of the design causes it. Then end with the complete corrected Verilog module in one code block.'
    else:
        msg += '\nFix the problems and return only the complete corrected Verilog module in one code block.'
    return [{'role': 'user', 'content': msg}]

def _legacy_v1_prompt(spec, history):
    msg = spec
    if history:
        code, score, fb = history[-1]
        msg += f'\n\nYour previous attempt:\n```verilog\n{code}\n```\nResult (score {score:.2f}):\n{fb}\n'
        if len(history) > 1:
            msg += '\nEarlier feedback you already received:\n' + '\n'.join((f'- {h[2].splitlines()[0]}' for h in history[:-1]))
        msg += '\nFix the problem and return the complete corrected module.'
        if len(history) > 1 and history[-1][0].strip() == history[-2][0].strip():
            msg += '\nYour last two attempts were identical and both failed. Change the logic, not just the formatting.\n'
    return [{'role': 'user', 'content': msg}]

def spec_path(path):
    """Resolve explicit specs and preserve aliases for the bundled specs we moved."""
    candidate = pathlib.Path(path)
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute() and candidate.parent != pathlib.Path('.'):
        return candidate.resolve()
    if candidate.is_absolute() and candidate.parent != ROOT:
        return candidate
    return next((p for p in (ROOT / 'prompts/specs').glob('*.md') if p.name.lower() == candidate.name.lower()), candidate.resolve())

def public_spec(problem, *, hint=False, interface_from_tb=False):
    text = spec_path(problem['spec']).read_text()
    text += '\n' + interface_line(problem, generated=interface_from_tb)
    if problem.get('spec_addendum'):
        text += '\n' + problem['spec_addendum']
    return text + (EXTRA_HINT if hint else '')

def _find_spec(name):
    return next((p for directory in (ROOT / 'prompts/specs', ROOT) for p in directory.glob('*.md') if p.stem.lower() == name.lower()), ROOT / (name + '.md'))

def problem_spec(problem, style='mine', hint=False):
    data = load_problem(_find_spec(problem))
    return public_spec(data) + (EXTRA_HINT if hint else '')

def interface_line(problem, *, generated=False):
    """The default formatting is unchanged; TB-derived formatting is opt-in."""
    p = _problem(problem)
    ports, module = (p['ports'], p['module'])
    if generated:
        if not p.get('official'):
            raise ValueError('--interface-from-tb requires a testbench')
        ref = p.get('ref_verilog') or next((path for path in p.get('references', ()) if pathlib.Path(path).exists()), None)
        if not ref:
            raise ValueError('--interface-from-tb requires an independent reference')
        parsed = file_problem(p['spec'], p['official'], ref)
        ports, module = (parsed['ports'], parsed['module'])
    interface = ', '.join((f'{direction} {name}' + (f' [{width - 1}:0]' if width > 1 else '') for name, direction, width in ports))
    return f'Use module name {module} with ports: {interface}.'

def prompt_problem(problem, *, spec=None, layout='bounded', **options):
    """Attach checker-owned prompt options without changing the grading data."""
    p = dict(_problem(problem))
    p['_prompt_options'] = dict(options, layout=layout)
    p['_prompt_spec'] = spec if spec is not None else public_spec(p, hint=options.get('hint', False), interface_from_tb=options.get('interface_from_tb', False))
    return p

def build_prompt(problem, history, explain=False, show_code=True):
    """Build checker-owned messages, including live KB lookup.

    Historical layouts preserve archived experiments verbatim. New Markdown runs
    use bounded context and show principles alone unless detect text is requested.
    """
    p = _problem(problem)
    options = p.get('_prompt_options', {})
    spec = p.get('_prompt_spec')
    if spec is None:
        spec = public_spec(p)
    explain = explain or options.get('explain', False)
    show_code = show_code and options.get('show_code', True)
    tuples = [(entry['code'], entry['score'], entry['feedback']) if isinstance(entry, dict) else tuple(entry) for entry in history]
    layout = options.get('layout', 'bounded')
    if layout == 'legacy':
        return _legacy_prompt(spec, tuples, show_code, explain)
    if layout == 'legacy_v1':
        return _legacy_v1_prompt(spec, tuples)
    input_budget = options.get('context_tokens', 8000) - options.get('max_tokens', 1500) - 512
    if input_budget < 512:
        raise ValueError('context budget must exceed output budget by at least 1024 tokens')
    taxonomy_path = pathlib.Path(options.get('taxonomy_path') or ROOT / 'taxonomy.json')
    knowledge_path = pathlib.Path(options.get('knowledge_path') or ROOT / 'knowledge.json')
    definitions, taxonomy_digest = taxonomy.snapshot(taxonomy_path)
    knowledge, knowledge_digest = taxonomy.snapshot(knowledge_path)
    excluded = options.get('exclude_categories', ())
    applicable = taxonomy.categories(spec, p['ports'], taxonomy_path, options.get('category_overrides', ()), data=definitions, exclude=excluded)
    diagnostics = history[-1].get('diagnostics', {}) if history and isinstance(history[-1], dict) else {}
    signatures = diagnostics.get('signatures', [])
    selected = taxonomy.select(taxonomy_path, applicable, signatures, limit=options.get('rules_limit', 2), data=definitions, knowledge=knowledge, show_detect_text=options.get('show_detect_text', False), max_bytes=min(options.get('rule_bytes', 1200), max(0, input_budget - _bytes(spec) - 700))) if options.get('principles', False) else []
    messages = _bounded_prompt(spec, tuples, show_code, explain, rules=selected, max_input_bytes=input_budget, show_detect_text=options.get('show_detect_text', False))
    p['_prompt_audit'] = dict(categories=applicable, signatures=signatures, selected_rules=selected, taxonomy_sha256=taxonomy_digest, knowledge_sha256=knowledge_digest, taxonomy_path=str(taxonomy_path), knowledge_path=str(knowledge_path), prompt_byte_budget=input_budget, prompt_utf8_bytes=prompt_size(messages), excluded_categories=list(excluded))
    return messages

def prompt_audit(problem):
    return dict(problem['_prompt_audit'])

def prompt_size(messages):
    return sum((_bytes(message['content']) for message in messages))

def feedback_options(problem, supplied):
    return dict(supplied, principles=False)

def no_module_result(diagnostics):
    diagnostics['signatures'] = []
    return (0.0, 'format: requirement: return a complete module. Observed: no module found.')

def validate_knowledge(taxonomy_path, knowledge_path):
    taxonomy.read(taxonomy_path)
    taxonomy.read(knowledge_path)

def save_prompt_event(record, source_log):
    """Store exact logged messages separately without changing the JSONL schema."""
    if not record.get('messages') or 'round' not in record:
        return
    source = pathlib.Path(source_log)
    run_name = source.parent.name if source.stem == 'events' else source.stem
    directory = ROOT / 'prompts/runs' / run_name
    directory.mkdir(parents=True, exist_ok=True)
    name = f"run_{record.get('run', 0)}_round_{record['round']}"
    (directory / (name + '.json')).write_text(json.dumps(record['messages'], ensure_ascii=False, indent=2) + '\n')
    (directory / (name + '.txt')).write_text('\n\n'.join((message['content'] for message in record['messages'])))
    (directory / 'source.json').write_text(json.dumps(dict(source_log=str(source)), indent=2) + '\n')

def run_logged(callback, argv=None):
    """Mirror plain console output to log/NAME.log; preserve original stdout."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    for option in ('problem', 'spec', 'tb', 'log-dir'):
        parser.add_argument('--' + option)
    parser.add_argument('--run-all', action='store_true')
    parser.add_argument('--knowledge-report', action='store_true')
    parser.add_argument('--no-console-log', action='store_true')
    options, _ = parser.parse_known_args(argv)
    if options.run_all or options.no_console_log or any((flag in (sys.argv[1:] if argv is None else argv) for flag in ('-h', '--help'))):
        return callback()
    name = options.problem or pathlib.Path(options.spec or options.tb or 'lfsr').stem.lower()
    if options.tb and (not options.problem) and (not options.spec):
        name = name.removesuffix('_tb')
    if options.knowledge_report:
        name = 'knowledge_report'
    name = re.sub('[^\\w.-]+', '_', name)
    directory = pathlib.Path(options.log_dir or ROOT / 'log')
    directory.mkdir(parents=True, exist_ok=True)

    class Tee:

        def __init__(self, console, stream):
            self.console, self.stream = (console, stream)

        def write(self, text):
            self.stream.write(text)
            return self.console.write(text)

        def flush(self):
            self.stream.flush()
            self.console.flush()
    with (directory / (name + '.log')).open('w') as stream, contextlib.redirect_stdout(Tee(sys.stdout, stream)), contextlib.redirect_stderr(Tee(sys.stderr, stream)):
        return callback()
BUILD = ROOT / 'build'
BUILD.mkdir(exist_ok=True)

def _principle_lines(v1_ids, v2_ids=(), *, principles=False, detectors=False, taxonomy_path=None, knowledge_path=None, categories=(), max_rules=2):
    """Share one two-line budget, with v1 taking priority over v2."""
    if not principles:
        return []
    ids = list(v1_ids) + (list(v2_ids) if detectors else [])
    if not ids:
        return []
    if taxonomy_path:
        definitions = taxonomy.read(taxonomy_path)
        knowledge = taxonomy.read(knowledge_path or pathlib.Path(__file__).with_name('knowledge.json'))
        return [taxonomy.entry_text(entry) for entry in taxonomy.select(taxonomy_path, categories, ids, limit=max_rules, data=definitions, knowledge=knowledge)]
    entries = json.loads(pathlib.Path(__file__).with_name('principles.json').read_text())
    principles = {entry['id']: entry['principle'] for entry in entries}
    lines = []
    for signature in dict.fromkeys(ids):
        line = 'Principle: ' + principles[signature]
        if line not in lines:
            lines.append(line)
        if len(lines) == 2:
            break
    return lines

def _with_principles(feedback, v1_ids, v2_ids=(), **options):
    return '\n'.join([feedback] + _principle_lines(v1_ids, v2_ids, **options))

def _lfsr_step(s):
    return s << 1 & 255 | (s >> 0 ^ s >> 3 ^ s >> 5 ^ s >> 6) & 1

def _sequence_next(state, inputs):
    position, value = state
    if not inputs['enable']:
        return state
    return ((position + 1) % len(SEQUENCE), SEQUENCE[position])
SEQUENCE = (175, 188, 226, 120, 255, 226, 11, 141)
RESET_REQUIREMENTS = {'reset_value': 'while {rst} is {act}, {out} must equal the initial state from the spec', 'reset_immediate': 'while {rst} is {act}, {out} must equal the initial state at every moment, including before the first clock edge', 'reset_midrun': 'reset during a run must restore the initial state {reset_timing}, hold it across a clock edge, and restart the specified sequence after release'}
PROBLEMS = {'lfsr': dict(module='lfsr', ports=(('clk', 'input', 1), ('reset_n', 'input', 1), ('data', 'output', 8)), clock='clk', reset='reset_n', reset_active=0, async_reset=True, model=dict(reset_state=138, next_state=lambda state, inputs: _lfsr_step(state), output=lambda state: {'data': state}), initial_inputs={}, reset_hold_cycles=1, restart_cycles=4, segments=[dict(label='sequence', cycles=256, inputs={}, first_check='first_step', requirement='every clock edge must advance {out} by one step of the LFSR in the spec', observed='at cycle {cycle} after release, {out} is {description}.')], requirements=dict(RESET_REQUIREMENTS, first_step='on each clock edge after {rst} is released, {out} must advance by one step of the LFSR in the spec'), observations=dict(first_step='on the first edge after release, {out} is {description}.'), weights=dict(compiles=0.1, reset_value=0.15, reset_immediate=0.15, reset_midrun=0.2, first_step=0.15, sequence=0.25), official='TestBench/lfsr_tb.v', spec='LFSR.md', references=('reference/lfsr.v', 'candidates/lfsr.v')), 'sequence_generator': dict(module='sequence_generator', ports=(('clk', 'input', 1), ('reset_n', 'input', 1), ('enable', 'input', 1), ('data', 'output', 8)), clock='clk', reset='reset_n', reset_active=0, async_reset=True, model=dict(reset_state=(1, SEQUENCE[0]), next_state=_sequence_next, output=lambda state: {'data': state[1]}), initial_inputs={'enable': 0}, reset_hold_cycles=2, restart_cycles=11, segments=[dict(label='sequence', cycles=8, inputs={'enable': 1}, first_check='first_step', requirement='enabled rising edges must advance through the specified sequence in order'), dict(label='repetition', cycles=16, inputs={'enable': 1}, requirement='the sequence must repeat in order across its boundary'), dict(label='disable', cycles=5, inputs={'enable': 0}, requirement='data and the sequence position must hold while enable is low'), dict(label='resume', cycles=16, inputs=[{'enable': 1}] * 3 + [{'enable': 0}] * 5 + [{'enable': 1}] * 8, requirement='enabling again must continue from the paused sequence position'), dict(label='reset_restart', cycles=11, inputs={'enable': 1}, reset_before={'enable': 1}, requirement='reset while enabled or disabled must restore the initial state and restart the sequence after release'), dict(label='reset_restart', cycles=11, inputs={'enable': 1}, reset_before={'enable': 0}, requirement='reset while enabled or disabled must restore the initial state and restart the sequence after release')], requirements=dict(RESET_REQUIREMENTS, first_step='the first enabled rising edge after reset must advance from the initial sequence item'), observations=dict(first_step='on the first enabled edge after release, {out} is {description}.'), weights=dict(compiles=0.1, reset_value=0.05, reset_immediate=0.05, reset_midrun=0.1, first_step=0.1, sequence=0.15, repetition=0.1, disable=0.1, resume=0.1, reset_restart=0.15), official='TestBench/sequence_generator_tb.v', spec='sequence_generator.md', references=('reference/seq/seq_a.v',))}
FAIL_RE = re.compile('^\\s*Error|completed with\\s+\\d+\\s+errors', re.I)
PASS_RE = re.compile('All test cases passed|Simulation successful|completed successfully', re.I)

def _problem(problem_name):
    if isinstance(problem_name, dict):
        return problem_name
    name = pathlib.Path(problem_name).stem
    if name.endswith('_tb'):
        name = name[:-3]
    return PROBLEMS[name]

def _probe(p):
    """Generate a probe and a private reference sample for every observation.

    All input assignments happen with the clock low. Every rising edge is sampled
    one ns later, including edges while reset is held. Reset assertions also get
    a sample before the next edge for asynchronous designs only.
    """
    clk, rst, act = (p['clock'] or '_checker_clock', p['reset'], p['reset_active'])
    outputs = [(name, width) for name, direction, width in p['ports'] if direction == 'output']
    inputs = dict(p['initial_inputs'])
    model = p.get('model') if not p.get('ref_verilog') else None
    state = model['reset_state'] if model else None
    reference = bool(p.get('ref_verilog'))
    lines = ['`timescale 1ns/1ps', 'module probe;']
    if not p['clock']:
        lines.append('  reg _checker_clock = 0;')
    for name, direction, width in p['ports']:
        size = f' [{width - 1}:0]' if width > 1 else ''
        if direction == 'input':
            value = 0 if name == clk else 1 - act if name == rst else inputs[name]
            lines.append(f'  reg{size} {name} = {value};')
        else:
            lines.append(f'  wire{size} {name};')
            if reference:
                lines.append(f'  wire{size} _checker_ref_{name};')
    connections = ', '.join((f'.{name}({name})' for name, _, _ in p['ports']))
    lines.append(f"  {p['module']} dut ({connections});")
    if reference:
        ref_connections = ', '.join((f".{name}({(name if direction == 'input' else '_checker_ref_' + name)})" for name, direction, _ in p['ports']))
        lines.append(f'  _checker_reference reference ({ref_connections});')
    lines.append('  initial begin')
    samples, now, edge_count = ([], 0, 0)
    transitions = {}

    def emit(text, delay=0):
        nonlocal now
        now += delay
        lines.append(f'    #{delay} {text}' if delay else f'    {text}')

    def drive(values):
        for name, value in values.items():
            if name not in inputs:
                raise ValueError(f'not a stimulus input: {name}')
            if inputs[name] != value:
                transitions[name] = (edge_count, value)
            inputs[name] = value
            emit(f'{name} = {value};')

    def sample(checks, when, *, kind='segment', cycle=0, subcheck=None, requirements=None, observed=None, previous=None, following=None):
        number = len(samples)
        names = [name for name, _ in outputs]
        if reference:
            names += ['_checker_ref_' + name for name, _ in outputs]
        fmt = ' '.join(('%b' for _ in names))
        args = ', '.join(names)
        emit(f'$display("P {number} {fmt}", {args});')
        context = [f"{edge_count - since} cycle{('s' if edge_count - since != 1 else '')} after {name} " + ('went low' if value == 0 else 'went high' if value == 1 else 'changed') for name, (since, value) in transitions.items()]
        samples.append(dict(checks=checks, want=model['output'](state) if model else {}, when=when, time=now, kind=kind, cycle=cycle, subcheck=subcheck, edge=edge_count, context=context, inputs=dict(inputs), requirements=requirements or {}, observed=observed, previous=model['output'](state if previous is None else previous) if model else {}, following=model['output'](state if following is None else following) if model else {}))

    def edge(checks, when, *, held=False, rise_delay=4, **metadata):
        nonlocal state, edge_count
        previous = state
        emit(f'{clk} = 1;', rise_delay)
        edge_count += 1
        if model:
            state = model['reset_state'] if held else model['next_state'](state, inputs)
        emit('', 1)
        following = (model['reset_state'] if held else model['next_state'](state, inputs)) if model else None
        sample(checks, when, previous=previous, following=following, **metadata)
        emit(f'{clk} = 0;', 4)

    def reset(check=None, initial=False, values=None):
        nonlocal state
        if values is not None:
            drive(values)
        if not rst:
            return
        emit(f'{rst} = {act};', 1)
        emit('', 1)
        if p['async_reset']:
            if model:
                state = model['reset_state']
            sample(('reset_immediate',) if initial else (check,), 'before the first clock edge' if initial else f'at {now} ns, one ns after reset assertion with {clk} low and no clock edge', kind='initial_immediate' if initial else 'reset', subcheck=0)
        for i in range(p['reset_hold_cycles']):
            edge(('reset_value',) if initial else (check,), f'at {now + (3 if i == 0 else 4) + 1} ns, after a clock edge with reset held', held=True, rise_delay=3 if i == 0 else 4, kind='initial_value' if initial else 'reset', subcheck=1)
        emit(f'{rst} = {1 - act};', 1)
    reset(initial=True)
    for segment in p['segments']:
        label = segment['label']
        requirement = {label: segment['requirement']}
        if 'reset_before' in segment:
            reset(label, values=segment['reset_before'])
        for i in range(segment['cycles']):
            values = segment['inputs']
            drive(values[i] if isinstance(values, (list, tuple)) else values)
            checks = (segment['first_check'], label) if i == 0 and 'first_check' in segment else (label,)
            edge(checks, f'at probe cycle {i + 1}', cycle=i + 1, requirements=requirement, observed=segment.get('observed'))
            emit('', 1)
    if not rst:
        lines.extend(['    $finish;', '  end', 'endmodule'])
        return ('\n'.join(lines) + '\n', samples)
    drive(p.get('midrun_inputs', p['segments'][-1]['inputs'] if isinstance(p['segments'][-1]['inputs'], dict) else inputs))
    if model:
        for _ in range(p.get('seek_limit', 256)):
            if state != model['reset_state']:
                break
            edge((), 'before midrun reset', kind='setup')
            emit('', 1)
        else:
            raise ValueError('reference never leaves the reset state')
    reset('reset_midrun')
    for i in range(1, p['restart_cycles'] + 1):
        edge(('reset_midrun',), f'at restart cycle {i} after reset release', kind='restart', cycle=i, subcheck=i + 1)
        emit('', 1)
    lines.extend(['    $finish;', '  end', 'endmodule'])
    return ('\n'.join(lines) + '\n', samples)

def _harness(p):
    return _probe(p)[0]

def _without_comments(source):
    return re.sub('//[^\\n]*|/\\*[\\s\\S]*?\\*/', '', source)

def file_problem(spec, tb, ref, *, seed=0, async_reset=True, cycles=256, random_cycles=128):
    """Infer the interface from a named testbench instantiation, then add stimulus.

    Benchmark testbenches declare driven signals as reg/logic and DUT outputs as
    wire. A logic signal is an input only if the testbench assigns it. Parameter
    widths may use integer localparams; unsupported connections fail explicitly.
    """
    if cycles < 1 or random_cycles < 1:
        raise ValueError('stimulus cycle counts must be positive')
    source = _without_comments(pathlib.Path(tb).read_text())
    spec = spec_path(spec)
    spec_text = spec.read_text()
    constants = {name: int(value) for name, value in re.findall('\\b(?:localparam|parameter)\\s+(?:integer\\s+)?(\\w+)\\s*=\\s*(\\d+)\\s*;', source)}

    def bound(expression):
        expression = expression.strip()
        if expression.isdecimal():
            return int(expression)
        if expression in constants:
            return constants[expression]
        match = re.fullmatch('(\\w+)\\s*([+-])\\s*(\\d+)', expression)
        if match and match[1] in constants:
            return constants[match[1]] + (1 if match[2] == '+' else -1) * int(match[3])
        raise ValueError(f'unsupported testbench port width: {expression}')
    signals = {}
    for kind, size, declarations in re.findall('\\b(reg|wire|logic)\\s*(?:signed\\s*)?(\\[[^\\]]+\\])?\\s*([^;]+);', source):
        width = 1
        if size:
            msb, lsb = size[1:-1].split(':')
            width = abs(bound(msb) - bound(lsb)) + 1
        for declaration in declarations.split(','):
            match = re.match('\\s*(\\w+)\\b', declaration)
            if match:
                signals[match[1]] = (kind, width)
    candidates = []
    for module, instance, connections in re.findall('\\b(\\w+)\\s+(?:#\\s*\\([^;]*?\\)\\s*)?(\\w+)\\s*\\(([^;]*?)\\)\\s*;', source):
        if module in ('module', 'if', 'for', 'while', 'task', 'function'):
            continue
        named = re.findall('\\.(\\w+)\\s*\\(\\s*([^()]*)\\s*\\)', connections)
        if named:
            candidates.append((module, instance, named))
    ref_source = _without_comments(pathlib.Path(ref).read_text())
    ref_modules = re.findall('\\bmodule\\s+(\\w+)', ref_source)
    matching = [candidate for candidate in candidates if candidate[0] in ref_modules]
    candidates = matching or candidates
    if len(candidates) != 1:
        raise ValueError('testbench must identify one DUT with named, signal-connected ports')
    module, _, connections = candidates[0]
    ports = []
    for port, signal in connections:
        signal = signal.strip()
        if signal not in signals:
            raise ValueError(f'testbench port {port} must connect to a declared signal')
        kind, width = signals[signal]
        driven = kind == 'reg' or (kind == 'logic' and re.search('\\b' + re.escape(signal) + '\\s*(?:<=|=(?!=))', source))
        ports.append((port, 'input' if driven else 'output', width))
    input_names = [name for name, direction, _ in ports if direction == 'input']
    clocks = [name for name in input_names if re.search('clk|clock', name, re.I)]
    resets = [name for name in input_names if re.search('reset|rst|clear', name, re.I)]
    if len(clocks) > 1 or len(resets) > 1:
        raise ValueError('multiple clock/reset inputs need an explicit interface')
    clock, reset = (clocks[0] if clocks else None, resets[0] if resets else None)
    async_reset = async_reset and reset is not None
    active = 0 if re.search('active[ -]*low', spec_text, re.I) or re.search('_n$|^n_?reset|^n_?rst', reset or '', re.I) else 1
    stimulus_ports = [(name, width) for name, direction, width in ports if direction == 'input' and name not in (clock, reset)]
    high = {name: (1 << width) - 1 for name, width in stimulus_ports}
    rng = random.Random(seed)
    segments = [dict(label='sequence', cycles=cycles, inputs=high, first_check='first_step', requirement='every output must follow the specified behavior on each rising edge'), dict(label='random', cycles=random_cycles, inputs=[{name: rng.getrandbits(width) for name, width in stimulus_ports} for _ in range(random_cycles)], requirement='every output must follow the specified behavior for the sampled inputs')]
    for name, _ in stimulus_ports:
        for value, cycles in ((0, 5), (high[name], 12)):
            segments.append(dict(label='toggles', cycles=cycles, inputs=dict(high, **{name: value}), requirement='input changes and holds must follow the specified behavior'))
    for enabled in (high, dict.fromkeys(high, 0)) if reset else ():
        segments.append(dict(label='reset_restart', cycles=11, inputs=high, reset_before=enabled, requirement='reset with either input level must restore and restart the specified behavior'))
    weights = dict(compiles=0.1, reset_value=0.1)
    if async_reset:
        weights['reset_immediate'] = 0.1
    else:
        weights['reset_value'] += 0.1
    weights.update(reset_midrun=0.1, first_step=0.1, sequence=0.15, random=0.1, toggles=0.15, reset_restart=0.1)
    if not reset:
        for key in ('reset_value', 'reset_immediate', 'reset_midrun', 'reset_restart'):
            weights['sequence'] += weights.pop(key, 0)
    if not stimulus_ports:
        weights['random'] += weights.pop('toggles')
    return dict(module=module, ports=tuple(ports), clock=clock, reset=reset, reset_active=active, async_reset=async_reset, ref_verilog=str(ref), spec=str(spec), official=str(tb), initial_inputs=dict.fromkeys(high, 0), midrun_inputs=high, reset_hold_cycles=2, restart_cycles=11, segments=segments, requirements=dict(RESET_REQUIREMENTS, first_step='the first rising edge after reset must follow the specified behavior'), observations={}, weights=weights, include_context=True, seed=seed)

def _reference_copy(p, tag):
    """Give the reference top a private name so identical DUT module names coexist."""
    source = _without_comments(pathlib.Path(p['ref_verilog']).read_text())
    modules = re.findall('\\bmodule\\s+(\\w+)', source)
    if not modules:
        raise ValueError('reference must declare a module')
    name = p.get('ref_module', p['module'] if p['module'] in modules else modules[0])
    source = re.sub('(\\bmodule\\s+)' + re.escape(name) + '\\b', '\\1_checker_reference', source, count=1)
    path = BUILD / f'{tag}_reference.v'
    path.write_text(source)
    return path

def _interface_testbench(ref):
    """Read simple ANSI or classic module ports when no official TB is supplied."""
    source = _without_comments(pathlib.Path(ref).read_text())
    match = re.search('\\bmodule\\s+(\\w+)\\s*\\((.*?)\\)\\s*;', source, re.S)
    if not match:
        raise ValueError('provide --tb for a reference with a parameterized or complex interface')
    module, header = match.groups()
    ports = []
    direction, width = (None, '')
    if re.search('\\b(input|output)\\b', header):
        for item in header.split(','):
            declared = re.match('\\s*(input|output)\\s+(?:(?:wire|reg|logic|signed)\\s+)*(\\[[^]]+\\])?\\s*(\\w+)\\s*$', item)
            if declared:
                direction, width, name = declared.groups()
                width = width or ''
            else:
                name = item.strip()
                if not direction or not re.fullmatch('\\w+', name):
                    raise ValueError('provide --tb for this reference interface')
            ports.append((name, direction, width))
    else:
        for direction, width, names in re.findall('\\b(input|output)\\s+(?:(?:wire|reg|logic|signed)\\s+)*(\\[[^]]+\\])?\\s*([^;]+);', source):
            ports.extend(((name.strip(), direction, width or '') for name in names.split(',')))
    if not ports:
        raise ValueError('no reference ports found; provide --tb')
    lines = ['module interface_description;']
    lines += [f"{('reg' if direction == 'input' else 'wire')} {width} {name};" for name, direction, width in ports]
    lines += [f'{module} dut (' + ', '.join((f'.{name}({name})' for name, _, _ in ports)) + ');', 'endmodule']
    path = BUILD / f'{module}_interface.v'
    path.write_text('\n'.join(lines) + '\n')
    return path

def load_problem(spec, tb=None, ref=None, *, seed=0, async_reset=None, cycles=256, random_cycles=128):
    """Load an arbitrary Markdown specification with its independent oracle files.

    Same-stem files are discovered automatically; explicit paths always win.
    Adjacent reference JSON holds public clarifications and optional contracts.
    No problem-specific Python branch or registry entry is required.
    """
    spec = spec_path(spec)
    if not spec.is_file():
        raise ValueError(f'specification does not exist: {spec}')
    stems = list(dict.fromkeys((spec.stem, spec.stem.lower())))
    root = pathlib.Path(__file__).resolve().parent
    if not ref:
        candidates = [directory / (stem + '.v') for directory in (spec.parent, spec.parent / 'reference', root / 'verilog/reference', root / 'reference') for stem in stems]
        ref = next((path for path in candidates if path.is_file()), None)
    if not ref:
        raise ValueError('a Markdown spec needs an independent reference: provide --ref or a same-stem reference file')
    ref = pathlib.Path(ref).resolve()
    metadata_path = ref.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if not tb:
        candidates = [directory / (stem + '_tb.v') for directory in (spec.parent / 'TestBench', root / 'TestBench') for stem in stems]
        tb = next((path for path in candidates if path.is_file()), None)
    official = str(pathlib.Path(tb).resolve()) if tb else None
    interface = tb or _interface_testbench(ref)
    timing = metadata.get('async_reset', True) if async_reset is None else async_reset
    p = file_problem(spec, interface, ref, seed=seed, async_reset=timing, cycles=cycles, random_cycles=random_cycles)
    if async_reset is True and (not p['reset']):
        raise ValueError('--async-reset requires a reset port')
    p['official'] = official
    p['spec_addendum'] = metadata.get('spec_addendum', '')
    p['comparisons'] = metadata.get('comparisons', {})
    return p

def _contract_description(contract, got, width, sample, previous, previous_sample):
    """Output contracts allow valid nondeterministic results without comparing PRNGs."""
    if len(got) != width:
        return 'not observed (the simulation ended early)'
    if set(got.lower()) - set('01'):
        return 'undefined (X)'
    in_reset = sample['kind'] in ('initial_value', 'initial_immediate', 'reset')
    if in_reset and contract.get('reset') == 'defined':
        return None
    trigger = contract.get('trigger')
    triggered = True
    if trigger:
        name = trigger['input']
        old = previous_sample['inputs'].get(name, 0) if previous_sample and previous_sample['kind'] not in ('initial_value', 'initial_immediate', 'reset') else 0
        new = sample['inputs'].get(name, 0)
        triggered = bool(new) and (trigger.get('edge') != 'rising' or not old)
    if not triggered and contract.get('hold_outside_trigger'):
        if len(previous) == width and (not set(previous) - set('01')):
            return _describe(got, int(previous, 2), width)
        return 'not observed (the previous sample is unavailable)'
    if contract['kind'] == 'range':
        selector = str(sample['inputs'][contract['selector']])
        low, high = contract['ranges'][selector]
        if low <= int(got, 2) <= high:
            return None
        return 'outside the permitted range for the sampled inputs'
    raise ValueError(f"unknown output contract: {contract['kind']}")

def _describe(got, want, w):
    """None if equal; otherwise what is wrong, never the expected value."""
    if len(got) != w:
        return 'not observed (the simulation ended early)'
    if set(got.lower()) & set('xz'):
        return 'undefined (X)'
    exp = format(want, f'0{w}b')
    bad = [w - 1 - i for i in range(w) if got[i] != exp[i]]
    if not bad:
        return None
    return f'wrong in bit(s) {bad}; the other {w - len(bad)} bits are correct'
V2_OBSERVATIONS = {'PARTIAL_X': 'some output bits are known while others are X or Z.', 'OUTPUT_STUCK': 'the output is unchanged from the previous observed state.', 'OUTPUT_INVERTED': 'every output bit is the opposite of the corresponding specified bit.', 'BIT_ORDER_REVERSED': 'the output matches the specified state with its bit order reversed.', 'SHIFTED_TOWARD_LSB': 'compared with the previous state, the bits moved one position toward bit 0.', 'SHIFTED_TOWARD_MSB': 'compared with the previous state, the bits moved one position toward the highest bit.', 'OFF_BY_ONE_VALUE': 'the unsigned output value differs from the specified value by exactly one.', 'ONE_CYCLE_LATE': 'the output matches the specified state from one cycle earlier.', 'ONE_CYCLE_EARLY': 'the output matches the specified state from one cycle later.', 'UPPER_BITS_ZERO': 'the upper {upper} output bits are zero while the lower {lower} bits match the specified state.'}

def _v2_signatures(got, want, previous, width, previous_expected, next_expected, *, detectors=True):
    """Compare only a failing sample; reference values never enter feedback.

    O = observed unsigned value, E = expected value, P = previous observed value.
    Partial X requires both known and X/Z bits; all other matches require known O.
    Stuck: O == P. Inverted: O == (~E & mask). Reversed: O == reverse_bits(E).
    Shifts compare all but the newly entered bit of O with the corresponding P bits.
    Off by one: abs(O - E) == 1 (no wrap). Late/early compare adjacent reference states.
    Upper zero: O's upper half is zero, E's upper half is nonzero, lower halves match.
    For odd widths, the lower half has width // 2 bits. Shifts require width > 1.
    """
    if not detectors or width < 1:
        return []
    got, previous = (got.lower(), previous.lower())
    if len(got) != width or set(got) - set('01xz'):
        return []
    if set(got) & set('xz'):
        return ['PARTIAL_X'] if set(got) & set('01') else []
    observed = int(got, 2)
    if observed == want:
        return []
    previous_known = len(previous) == width and (not set(previous) - set('01'))
    mask = (1 << width) - 1
    lower = width // 2
    lower_mask = (1 << lower) - 1
    matches = {'OUTPUT_STUCK': previous_known and got == previous, 'OUTPUT_INVERTED': observed == ~want & mask, 'BIT_ORDER_REVERSED': got == format(want, f'0{width}b')[::-1], 'SHIFTED_TOWARD_LSB': width > 1 and previous_known and (got[1:] == previous[:-1]), 'SHIFTED_TOWARD_MSB': width > 1 and previous_known and (got[:-1] == previous[1:]), 'OFF_BY_ONE_VALUE': abs(observed - want) == 1, 'ONE_CYCLE_LATE': observed == previous_expected, 'ONE_CYCLE_EARLY': observed == next_expected, 'UPPER_BITS_ZERO': lower > 0 and observed >> lower == 0 and (want >> lower != 0) and (observed & lower_mask == want & lower_mask)}
    return [signature for signature in V2_OBSERVATIONS if matches.get(signature, False)]

def _v2_observation(signature, width):
    return 'Observed pattern: ' + V2_OBSERVATIONS[signature].format(upper=width - width // 2, lower=width // 2)

def _official_ok(tb_path, dut_path, tag, timeout):
    """True/False from the benchmark's own testbench; None if it is not present."""
    if not tb_path or not pathlib.Path(tb_path).exists():
        return None
    exe = BUILD / f'{tag}_official.vvp'
    c = subprocess.run(['iverilog', '-g2012', '-I', str(pathlib.Path(tb_path).resolve().parent), '-I', str(pathlib.Path(dut_path).resolve().parent), '-o', str(exe), str(tb_path), str(dut_path)], capture_output=True, text=True)
    if c.returncode != 0:
        return False
    try:
        r = subprocess.run(['vvp', str(exe.resolve())], cwd=BUILD, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    lines = r.stdout.splitlines()
    if any((FAIL_RE.search(l) for l in lines)):
        return False
    return any((PASS_RE.search(l) for l in lines))

def official(problem_name, dut_path, timeout=30):
    p = _problem(problem_name)
    return _official_ok(p['official'], dut_path, pathlib.Path(dut_path).stem, timeout)

def code_facts(dut_path, out):
    """Observe procedural drivers in comment-free source, preserving source lines."""
    source = pathlib.Path(dut_path).read_text()
    source = re.sub('//[^\\n]*|/\\*[\\s\\S]*?\\*/', lambda m: re.sub('[^\\n]', ' ', m.group()), source)
    tokens = list(re.finditer('"(?:\\\\.|[^"\\\\])*"|[A-Za-z_$][\\w$]*|\\S', source))

    def statement(i):
        if i >= len(tokens):
            return i
        word = tokens[i].group()
        if word == 'begin':
            i += 1
            if i < len(tokens) and tokens[i].group() == ':':
                i += 2
            while i < len(tokens) and tokens[i].group() != 'end':
                i = statement(i)
            return min(i + 1, len(tokens))
        if word in ('if', 'for', 'while', 'repeat'):
            i += 1
            depth = 0
            while i < len(tokens):
                t = tokens[i].group()
                depth += (t == '(') - (t == ')')
                i += 1
                if t == ')' and depth == 0:
                    break
            i = statement(i)
            if word == 'if' and i < len(tokens) and (tokens[i].group() == 'else'):
                i = statement(i + 1)
            return i
        if word in ('case', 'casex', 'casez'):
            depth = 1
            i += 1
            while i < len(tokens) and depth:
                t = tokens[i].group()
                depth += (t in ('case', 'casex', 'casez')) - (t == 'endcase')
                i += 1
            return i
        while i < len(tokens):
            t = tokens[i].group()
            i += 1
            if t == ';':
                break
        return i
    assignment = re.compile('(?<![\\w$])' + re.escape(out) + '(?![\\w$])\\s*(?:\\[[^\\]]+\\]\\s*)?(<=|=(?!=))')
    blocks = []
    for match in re.finditer('\\balways\\s*@\\s*\\(([^)]*)\\)|\\binitial\\b', source):
        i = next((j for j, t in enumerate(tokens) if t.start() >= match.end()), len(tokens))
        end = statement(i)
        body_end = tokens[end - 1].end() if end > i else match.end()
        assignments = list(assignment.finditer(source, match.end(), body_end))
        if assignments:
            blocks.append(dict(line=source.count('\n', 0, match.start()) + 1, kind='always' if match.group(1) is not None else 'initial', sensitivity=match.group(1), blocking_lines=[source.count('\n', 0, a.start()) + 1 for a in assignments if a.group(1) == '=']))
    return dict(blocks=blocks, multiple_blocks=len(blocks) > 1, blocking_lines=sorted({line for b in blocks for line in b['blocking_lines']}))

def _facts_line(out, facts):
    places = [f"line {b['line']}, " + (f"a block triggered by {b['sensitivity']}" if b['kind'] == 'always' else 'an initial block') for b in facts['blocks']]
    line = f'In your design, {out} is assigned in {len(places)} place(s): ' + '; '.join(places) + '.'
    if facts['multiple_blocks']:
        line += f' ({out} is driven from more than one block)'
    for number in facts['blocking_lines']:
        line += f' ({out} uses a blocking assignment on line {number})'
    return line

def _v1_signatures(parts, rows, facts, midrun_failure):
    """Select existing principles from failed checks and observed source facts."""
    before_edge_x = bool(set(rows.get('R0', '').lower()) & set('xz'))
    clocked_blocking = any((b['blocking_lines'] and b['kind'] == 'always' and re.search('\\b(?:posedge|negedge)\\b', b['sensitivity']) for b in facts['blocks']))
    matches = {'X_BEFORE_FIRST_EDGE': not parts['reset_immediate'] and before_edge_x, 'WRONG_BETWEEN_EDGES_IN_RESET': not parts['reset_immediate'] and (not before_edge_x) or midrun_failure == 0, 'WRONG_AFTER_EDGE_IN_RESET': not parts['reset_value'] or midrun_failure == 1, 'MULTIPLE_DRIVERS': facts['multiple_blocks'], 'BLOCKING_IN_CLOCKED_LOGIC': clocked_blocking, 'INIT_ONLY_VALUE': midrun_failure == 0 and any((b['kind'] == 'initial' for b in facts['blocks'])), 'NEXT_STATE_WRONG_FIRST_EDGE': not parts['first_step'], 'NEXT_STATE_DIVERGES_LATER': parts['first_step'] and (not parts['sequence'])}
    return [signature for signature, matched in matches.items() if matched]

def grade(problem_name, dut_path, timeout=30, *, facts=True, detectors=False, principles=False, async_reset=None, diagnostics=None, taxonomy_path=None, knowledge_path=None, categories=(), max_rules=2):
    p = copy.deepcopy(_problem(problem_name))
    if async_reset is not None and async_reset != p['async_reset']:
        p['async_reset'] = async_reset
        if not async_reset:
            p['weights']['reset_value'] += p['weights'].pop('reset_immediate', 0)
        else:
            share = p['weights']['reset_value'] / 2
            p['weights']['reset_value'] -= share
            p['weights']['reset_immediate'] = share
    options = dict(principles=principles, detectors=detectors, taxonomy_path=taxonomy_path, knowledge_path=knowledge_path, categories=categories, max_rules=max_rules)
    if principles and taxonomy_path and (diagnostics is None):
        diagnostics = {}
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(signatures=[], checks={}, mismatches=[], categories=list(categories))
    weights = p['weights']
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise ValueError('problem weights must sum to 1.0')
    if not p['async_reset'] and 'reset_immediate' in weights:
        raise ValueError('synchronous reset must not weight reset_immediate')
    tag = pathlib.Path(dut_path).stem
    source, samples = _probe(p)
    probe = BUILD / f'{tag}_probe.v'
    probe.write_text(source)
    exe = BUILD / f'{tag}_probe.vvp'
    sources = [str(probe), str(dut_path)]
    if p.get('ref_verilog'):
        sources.append(str(_reference_copy(p, tag)))
    include_dirs = ['-I', str(pathlib.Path(dut_path).resolve().parent)]
    if p.get('ref_verilog'):
        include_dirs += ['-I', str(pathlib.Path(p['ref_verilog']).resolve().parent)]
    c = subprocess.run(['iverilog', '-g2012', *include_dirs, '-s', 'probe', '-o', str(exe), *sources], capture_output=True, text=True)
    outputs = [(name, width) for name, direction, width in p['ports'] if direction == 'output']
    if c.returncode != 0:
        ports = [name + (f'[{width - 1}:0]' if width > 1 else '') for name, _, width in p['ports']]
        port_text = ', '.join(ports[:-1]) + ' and ' + ports[-1]
        feedback = f"compiles: requirement: a module named `{p['module']}` with ports {port_text} that compiles. Observed: compiler output:\n" + c.stderr.strip()[:1200]
        if diagnostics is not None:
            diagnostics['signatures'] = ['COMPILE_ERROR']
        return (0.0, _with_principles(feedback, ['COMPILE_ERROR'], **options))
    try:
        r = subprocess.run(['vvp', str(exe)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        feedback = 'timeout: requirement: the simulation must finish. Observed: it never finished.'
        if diagnostics is not None:
            diagnostics['signatures'] = ['TIMEOUT']
        return (weights['compiles'], _with_principles(feedback, ['TIMEOUT'], **options))
    rows, ref_rows = ({}, {})
    reference = bool(p.get('ref_verilog'))
    count = len(outputs)
    for line in r.stdout.splitlines():
        fields = line.split()
        if len(fields) == count * (2 if reference else 1) + 2 and fields[0] == 'P' and fields[1].isdigit():
            number = int(fields[1])
            rows[number] = dict(zip((name for name, _ in outputs), fields[2:2 + count]))
            if reference:
                values = fields[2 + count:]
                if any((len(value) != width or set(value) - set('01') for value, (_, width) in zip(values, outputs))):
                    return (0.0, 'reference: requirement: reference outputs must be defined. Observed: the reference produced an undefined or invalid sample.')
                ref_rows[number] = dict(zip((name for name, _ in outputs), (int(value, 2) for value in values)))
    parts = dict.fromkeys(weights, True)
    failures = {}
    fact_data = {name: code_facts(dut_path, name) for name, _ in outputs}
    v2_ids, pattern_lines = ([], [])
    midrun_failure = None
    for number, sample in enumerate(samples):
        for out, width in outputs:
            got = rows.get(number, {}).get(out, '')
            want = ref_rows.get(number, {}).get(out, 0) if reference else sample['want'][out]
            comparison = p.get('comparisons', {}).get(out)
            description = _contract_description(comparison, got, width, sample, rows.get(number - 1, {}).get(out, ''), samples[number - 1] if number else None) if comparison else _describe(got, want, width)
            if description is None:
                continue
            new_checks = [check for check in sample['checks'] if check not in failures]
            for check in new_checks:
                parts[check] = False
                context = dict(rst=p['reset'], act=p['reset_active'], out=out, cycle=sample['cycle'], description=description, reset_timing='immediately' if p['async_reset'] else 'across a clock edge')
                requirement = sample['requirements'].get(check, p['requirements'].get(check))
                if requirement is None:
                    requirement = next((s['requirement'] for s in p['segments'] if s['label'] == check))
                if sample['kind'] == 'initial_value':
                    observed = f"with {p['reset']} held at {p['reset_active']}, after a clock edge, {out} is {description}."
                elif sample['kind'] == 'initial_immediate':
                    observed = f"at {sample['time']} ns, with {p['reset']} at {p['reset_active']} and no clock edge yet, {out} is {description}."
                elif sample['kind'] == 'restart':
                    observed = f"reset restored and held the initial state correctly; the sequence after restart fails {sample['when']}, {out} is {description}."
                else:
                    template = p['observations'].get(check, sample['observed'])
                    observed = template.format(**context) if template else f"{sample['when']}, {out} is {description}."
                if p.get('include_context'):
                    observed = f"at cycle {sample['edge']}, {out} is {description}."
                    if sample['context']:
                        observed += ' Context: ' + '; '.join(sample['context']) + '.'
                message = f'{check}: requirement: {requirement.format(**context)}. Observed: {observed}'
                if facts and sample['kind'] in ('initial_value', 'initial_immediate', 'reset'):
                    message += '\n' + _facts_line(out, fact_data[out])
                failures[check] = message
                if diagnostics is not None:
                    diagnostics['mismatches'].append(dict(check=check, output=out, width=width, cycle=sample['edge'], description=description, context=sample['context'], inputs=sample['inputs'], kind=sample['kind']))
                if check == 'reset_midrun':
                    midrun_failure = sample['subcheck']
            if new_checks and sample['kind'] == 'segment' and (detectors or diagnostics is not None) and (not comparison):
                previous_expected = ref_rows.get(number - 1, {}).get(out) if reference else sample['previous'][out]
                next_edge = next((i for i in range(number + 1, len(samples)) if samples[i]['edge'] > sample['edge']), None)
                next_expected = ref_rows.get(next_edge, {}).get(out) if reference else sample['following'][out]
                ids = _v2_signatures(got, want, rows.get(number - 1, {}).get(out, ''), width, previous_expected, next_expected)
                v2_ids.extend(ids)
                if detectors:
                    pattern_lines.extend((_v2_observation(signature, width) for signature in ids))
    score = round(sum((weight for check, weight in weights.items() if parts[check])), 3)
    if diagnostics is not None:
        diagnostics['checks'] = parts.copy()
    if all(parts.values()):
        if _official_ok(p['official'], dut_path, tag, timeout) is False:
            return (0.9, 'official: requirement: the benchmark testbench must pass. Observed: the benchmark testbench failed.')
        return (1.0, 'correct')
    v1_ids = []
    if principles or diagnostics is not None:
        signature_parts = dict(parts)
        signature_parts.setdefault('reset_immediate', True)
        signature_parts.setdefault('reset_value', True)
        signature_parts.setdefault('first_step', True)
        signature_parts['sequence'] = all((parts[s['label']] for s in p['segments']))
        initial = next((i for i, sample in enumerate(samples) if sample['kind'] == 'initial_immediate'), None)
        for out, _ in outputs:
            initial_rows = {'R0': rows.get(initial, {}).get(out, '')}
            v1_ids.extend(_v1_signatures(signature_parts, initial_rows, fact_data[out], midrun_failure))
    semantic_ids = []
    if diagnostics is not None:
        for mismatch in diagnostics['mismatches']:
            if mismatch['description'] == 'outside the permitted range for the sampled inputs':
                semantic_ids.append('VALUE_OUT_OF_RANGE')
            if mismatch['kind'] == 'segment' and mismatch['inputs'].get('enable') == 0:
                semantic_ids.append('WRONG_WHILE_DISABLED')
            elif mismatch['kind'] == 'segment' and any(('enable went high' in context for context in mismatch['context'])) and any((s['kind'] == 'segment' and s['edge'] < mismatch['cycle'] and (s['inputs'].get('enable') == 0) for s in samples)):
                semantic_ids.append('WRONG_AFTER_REENABLE')
            if mismatch['check'] == 'repetition':
                semantic_ids.append('WRONG_AFTER_WRAP')
        diagnostics['signatures'] = list(dict.fromkeys(v1_ids + semantic_ids + v2_ids))
    messages = [failures[check] for check in weights if check in failures]
    return (score, _with_principles('\n'.join(messages + pattern_lines), v1_ids + semantic_ids, v2_ids, **options))

def run_suite(args):
    """Run the manifest in Python and preserve the existing suite log format."""
    manifest_path = pathlib.Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + str(time.time_ns())
    run_name = manifest.get('version', 'md_generic_dynamic_taxonomy') + '_' + stamp
    directory = args.log_dir.resolve() / 'runs' / run_name
    directory.mkdir(parents=True, exist_ok=False)
    tasks = [task for task in manifest['tasks'] if not args.only or task['name'] in args.only]
    if not tasks:
        raise ValueError('no tasks selected')
    if args.only and set(args.only) - {task['name'] for task in tasks}:
        raise ValueError('unknown task name in --only')
    (directory / 'run.json').write_text(json.dumps(dict(version=manifest.get('version'), mode=args.mode, seed=args.seed, manifest=manifest, manifest_path=str(manifest_path.resolve()), manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(), arguments={key: str(value) if isinstance(value, pathlib.Path) else value for key, value in vars(args).items()}), indent=2) + '\n')
    results = []
    for task in tasks:

        def path(value):
            candidate = pathlib.Path(value)
            return str(candidate if candidate.is_absolute() else ROOT / candidate)
        command = [sys.executable, str(ROOT / 'checker.py'), '--no-console-log', '--spec', path(task['spec']), '--seed', str(args.seed), '--cycles', str(manifest.get('cycles', 256)), '--random-cycles', str(manifest.get('random_cycles', 128)), '--taxonomy', path(manifest.get('taxonomy', 'taxonomy.json')), '--rules-limit', str(manifest.get('rules_limit', 2)), '--facts', '--detectors']
        if args.mode == 'agent':
            command.insert(2, '--agent')
        command += ['--knowledge', path(manifest.get('knowledge', 'knowledge.json'))]
        if args.show_detect_text:
            command += ['--show-detect-text']
        if args.exclude_categories:
            command += ['--exclude-categories', args.exclude_categories]
        for flag in ('tb', 'ref'):
            if task.get(flag):
                command += ['--' + flag, path(task[flag])]
        if not args.no_principles:
            command += ['--principles']
        if args.async_reset is not None:
            command += ['--async-reset' if args.async_reset else '--no-async-reset']
        for category in task.get('categories', []):
            command += ['--category', category]
        if args.mode == 'agent':
            command += ['--repeat', str(args.repeat), '--rounds', str(args.rounds), '--context-tokens', str(args.context_tokens if args.context_tokens is not None else manifest.get('context_tokens', 8000)), '--max-tokens', str(args.max_tokens if args.max_tokens is not None else manifest.get('max_tokens', 1500)), '--rule-bytes', str(args.rule_bytes if args.rule_bytes is not None else manifest.get('rule_bytes', 1200)), '--log-dir', str(directory), '--run-name', run_name, '--output-dir', str(ROOT / 'candidates/gen')]
            if not args.no_explain:
                command += ['--explain']
            for flag in ('interface_from_tb', 'no_code', 'hint'):
                if getattr(args, flag, False):
                    command += ['--' + flag.replace('_', '-')]
            if args.model:
                command += ['--model', args.model]
            if args.base_url:
                command += ['--base-url', args.base_url]
        else:
            candidate = args.dut_dir.resolve() / (task['name'] + '.v') if args.dut_dir else pathlib.Path(path(task['ref']))
            command += ['--dut', str(candidate), '--diagnostics-json', str(directory / (task['name'] + '.json'))]
        logfile = args.log_dir.resolve() / (task['name'] + '.log')
        previous_records = set(directory.rglob('*.jsonl'))
        print(f"{task['name']}: {logfile}", flush=True)
        with logfile.open('w') as stream:
            start = time.monotonic()
            completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        shutil.copyfile(logfile, directory / logfile.name)
        result = dict(task=task['name'], exit_code=completed.returncode, elapsed_seconds=time.monotonic() - start, log=str(logfile), command=command)
        record_files = sorted(set(directory.rglob('*.jsonl')) - previous_records)
        result['records'] = [str(path) for path in record_files]
        for record_file in record_files:
            for line in record_file.read_text().splitlines():
                record = json.loads(line)
                if record.get('event') == 'summary':
                    result['scores'] = record
        report = directory / (task['name'] + '.json')
        if args.mode == 'checker' and report.exists():
            result['score'] = json.loads(report.read_text())['score']
        results.append(result)
        (directory / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'Suite logs: {directory}', flush=True)
    return int(any((result['exit_code'] != 0 for result in results)))

def interface_comparison():
    rows = []
    for name in ('lfsr', 'sequence_generator'):
        problem = load_problem(_find_spec(name))
        rows.append(dict(problem=name, current=interface_line(problem), generated=interface_line(problem, generated=True)))
    return rows

def run_testbench(tb_path, dut_path, timeout=30):
    """Compile and run the provided benchmark without requiring a spec/reference."""
    if not pathlib.Path(tb_path).is_file() or not pathlib.Path(dut_path).is_file():
        raise ValueError('testbench and DUT must both be existing files')
    return _official_ok(tb_path, dut_path, pathlib.Path(dut_path).stem, timeout)

def _main(argv=None):
    import argparse, glob, sys
    parser = argparse.ArgumentParser(description='Grade a DUT or run checker selftests')
    parser.add_argument('--problem', choices=PROBLEMS)
    parser.add_argument('--dut')
    parser.add_argument('--spec')
    parser.add_argument('--tb')
    parser.add_argument('--ref')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cycles', type=int, default=256)
    parser.add_argument('--random-cycles', type=int, default=128)
    parser.add_argument('--taxonomy', default=str(pathlib.Path(__file__).with_name('taxonomy.json')))
    parser.add_argument('--knowledge', default=str(pathlib.Path(__file__).with_name('knowledge.json')))
    parser.add_argument('--show-detect-text', action='store_true')
    parser.add_argument('--knowledge-report', action='store_true', help='report every knowledge key difference without editing inputs')
    parser.add_argument('--knowledge-original', default=str(ROOT / 'knowledge_original.json'))
    parser.add_argument('--exclude-categories', default='')
    parser.add_argument('--category', action='append', default=[])
    parser.add_argument('--rules-limit', type=int, default=2)
    parser.add_argument('--diagnostics-json', help='write signatures and observations as JSON')
    parser.add_argument('--facts', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--detectors', action='store_true')
    parser.add_argument('--principles', action='store_true')
    reset = parser.add_mutually_exclusive_group()
    reset.add_argument('--async-reset', dest='async_reset', action='store_true')
    reset.add_argument('--no-async-reset', dest='async_reset', action='store_false')
    parser.set_defaults(async_reset=None)
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--run-all', action='store_true', help='run all tasks in the manifest using Python')
    parser.add_argument('--manifest', type=pathlib.Path, default=ROOT / 'experiments/md_generic_v4_dynamic_taxonomy.json')
    parser.add_argument('--mode', choices=('agent', 'checker'), default='checker')
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--dut-dir', type=pathlib.Path)
    parser.add_argument('--log-dir', type=pathlib.Path, default=ROOT / 'log')
    parser.add_argument('--no-console-log', action='store_true')
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--model')
    parser.add_argument('--base-url')
    parser.add_argument('--no-principles', action='store_true')
    parser.add_argument('--no-explain', action='store_true')
    parser.add_argument('--explain', action='store_true')
    parser.add_argument('--principles-v2', action='store_true')
    parser.add_argument('--hint', action='store_true')
    parser.add_argument('--no-code', action='store_true')
    parser.add_argument('--interface-from-tb', action='store_true')
    parser.add_argument('--show-interface-lines', action='store_true')
    parser.add_argument('--context-tokens', type=int)
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--rule-bytes', type=int)
    parser.add_argument('--run-tb', action='store_true', help='compile and run --tb with --dut, without a spec/reference')
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args(argv)
    if args.knowledge_report:
        report = taxonomy.knowledge_report(args.knowledge, args.knowledge_original)
        print(json.dumps(report, indent=2))
        return 0 if report['status'] == 'compared' else 1
    if args.run_all:
        if args.dut or args.spec or args.problem:
            parser.error('--run-all uses the manifest; use --only or --dut-dir to select tasks/candidates')
        try:
            return run_suite(args)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    if args.show_interface_lines:
        rows = interface_comparison()
        print(json.dumps(rows, indent=2))
        return 0
    if args.run_tb or (args.tb and args.dut and (not args.spec) and (not args.problem)):
        if not args.tb or not args.dut:
            parser.error('--run-tb requires --tb and --dut')
        try:
            passed = run_testbench(args.tb, args.dut, args.timeout)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print('correct' if passed else 'official: requirement: the benchmark testbench must pass. Observed: the benchmark testbench failed.')
        return 0 if passed else 1
    args.principles = args.principles or args.principles_v2
    args.detectors = args.detectors or args.principles_v2
    grade_options = dict(facts=args.facts, detectors=args.detectors, principles=args.principles, async_reset=args.async_reset)
    file_based = any((args.spec, args.tb, args.ref))
    if file_based:
        if args.problem or not args.spec:
            parser.error('use --spec FILE with optional --tb and --ref, or --problem')
        try:
            selected = load_problem(args.spec, args.tb, args.ref, seed=args.seed, async_reset=args.async_reset, cycles=args.cycles, random_cycles=args.random_cycles)
            if args.principles:
                applicable = taxonomy.categories(pathlib.Path(selected['spec']).read_text(), selected['ports'], args.taxonomy, args.category, exclude=[value.strip() for value in args.exclude_categories.split(',') if value.strip()])
                grade_options.update(taxonomy_path=args.taxonomy, knowledge_path=args.knowledge, categories=applicable, max_rules=args.rules_limit)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    else:
        selected = args.problem or 'lfsr'
    if args.dut:
        diagnostic_data = {} if args.diagnostics_json or (args.show_detect_text and args.principles) else None
        direct_options = dict(grade_options)
        if args.show_detect_text and args.principles:
            direct_options['principles'] = False
        score, feedback = grade(selected, args.dut, diagnostics=diagnostic_data, **direct_options)
        if args.show_detect_text and args.principles:
            p = _problem(selected)
            applicable = taxonomy.categories(public_spec(p), p['ports'], args.taxonomy, args.category, exclude=[value.strip() for value in args.exclude_categories.split(',') if value.strip()])
            entries = taxonomy.select(args.taxonomy, applicable, diagnostic_data['signatures'], knowledge=taxonomy.read(args.knowledge), limit=args.rules_limit, show_detect_text=True)
            if entries:
                feedback += '\n' + '\n'.join((taxonomy.entry_text(entry, show_detect_text=True) for entry in entries))
        print(f'score: {score}\n{feedback}')
        if args.diagnostics_json:
            report = pathlib.Path(args.diagnostics_json)
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(dict(score=score, feedback=feedback, diagnostics=diagnostic_data), indent=2) + '\n')
        if not args.selftest:
            sys.exit(0 if score == 1.0 else 1)
    if args.selftest and (not args.problem) and (not file_based):
        cases = [(name, p, p['references']) for name, p in PROBLEMS.items()]
        file_sequence = file_problem('sequence_generator.md', 'TestBench/sequence_generator_tb.v', 'reference/seq/seq_a.v', seed=args.seed)
        cases.append(('sequence_generator (files)', file_sequence, ('reference/seq/seq_a.v',)))
        rc = 0
        for name, p, references in cases:
            refs = [next((f for f in references if pathlib.Path(f).exists()))]
            pattern = 'candidates/bad/*.v' if p['module'] == 'lfsr' else f"candidates/bad/{p['module']}/*.v"
            for f in refs + sorted(glob.glob(pattern)):
                score, feedback = grade(p, f, **grade_options)
                print(f'{name}: {f}: {score}\n    ' + feedback.replace('\n', '\n    '))
                rc |= score != 1.0 if f in refs else score >= 1.0
        import unittest
        suite = _embedded_suite('test_*.py')
        rc |= not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
        print('\nSELFTEST ' + ('PASSED' if rc == 0 else 'FAILED'))
        sys.exit(rc)
    args.problem = args.problem or 'lfsr'
    if args.problem != 'lfsr' or file_based:
        p = selected if file_based else PROBLEMS[args.problem]
        references = (p['ref_verilog'],) if file_based else p['references']
        rc = 0
        for f in [*references, *sorted(glob.glob(f"candidates/bad/{p['module']}/*.v"))]:
            score, feedback = grade(p, f, **grade_options)
            print(f'{f}: {score}\n    ' + feedback.replace('\n', '\n    '))
            reference = f in references
            if reference:
                verdict = official(p, f)
                print(f'    Official testbench: {verdict}')
                rc |= score != 1.0 or verdict is not True
            else:
                rc |= score >= 1.0
        import unittest
        suite = _embedded_suite('test_engine.py')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            rc = 1
        print('\nSELFTEST ' + ('PASSED' if rc == 0 else 'FAILED'))
        sys.exit(rc)
    ref = next((f for f in ['reference/lfsr.v', 'candidates/lfsr.v'] if pathlib.Path(f).exists()), None)
    if ref is None:
        print('no reference design found (reference/lfsr.v or candidates/lfsr.v)')
        sys.exit(1)
    files = [ref] + sorted(glob.glob('candidates/bad/*.v'))
    expected_facts = {'lfsr_syncreset.v': 'In your design, data is assigned in 1 place(s): line 6, a block triggered by posedge clk.', 'lfsr_initial_hack.v': 'In your design, data is assigned in 2 place(s): line 6, an initial block; line 8, a block triggered by posedge clk. (data is driven from more than one block) (data uses a blocking assignment on line 6)', 'lfsr_two_drivers.v': 'In your design, data is assigned in 2 place(s): line 6, a block triggered by posedge clk; line 10, a block triggered by reset_n. (data is driven from more than one block)'}
    expected_scores = {'lfsr_badtap.v': 0.55, 'lfsr_initial_hack.v': 0.8, 'lfsr_syncreset.v': 0.65, 'lfsr_two_drivers.v': 0.65, 'shift_wrong_direction.v': 0.4, 'inverted_feedback.v': 0.4}
    rc = 0
    for f in files:
        s, fb = grade('lfsr', f, **grade_options)
        print(f'{f}: {s}\n    ' + fb.replace('\n', '\n    '))
        if f == ref and s != 1.0 or (f != ref and (not s < 1.0)):
            rc = 1
        if f != ref:
            name = pathlib.Path(f).name
            lines = fb.splitlines()
            reset_lines = [i for i, line in enumerate(lines) if line.startswith(('reset_value:', 'reset_immediate:', 'reset_midrun:'))]
            restart_lines = [i for i in reset_lines if 'reset restored and held the initial state correctly;' in lines[i]]
            facts_lines = [i for i in reset_lines if i not in restart_lines]
            expected = expected_facts.get(name, _facts_line('data', code_facts(f, 'data')))
            valid = bool(reset_lines) and all((i + 1 < len(lines) and lines[i + 1] == expected for i in facts_lines))
            valid = valid and sum((line.startswith('In your design,') for line in lines)) == len(facts_lines)
            if name == 'lfsr_badtap.v':
                valid = valid and len(restart_lines) == 1 and (not facts_lines)
            checks = [line.split(':', 1)[0] for line in lines if ': requirement:' in line]
            valid = valid and len(checks) == len(set(checks))
            valid = valid and (name not in expected_scores or s == expected_scores[name])
            print('    Facts and score ' + ('verified' if valid else 'FAILED'))
            if not valid:
                rc = 1
    import unittest
    suite = _embedded_suite('test_checker.py')
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        rc = 1
    print('\nSELFTEST ' + ('PASSED' if rc == 0 else 'FAILED'))
    sys.exit(rc)

# ---- model transport, iteration, and logging ----
"""Generic Markdown-to-Verilog iteration with bounded, dynamic taxonomy retrieval."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import time
import urllib.request
from datetime import datetime, timezone
AGENT_VERSION = 'md_generic_v4_dynamic_taxonomy'
MODEL_BASE = os.environ.get('KERNEL_AGENT_BASE_URL', 'http://localhost:8000/v1').rstrip('/')
MODEL_BASE = MODEL_BASE if MODEL_BASE.endswith('/v1') else MODEL_BASE + '/v1'

def http(path, body=None):
    req = urllib.request.Request(MODEL_BASE + path, data=json.dumps(body).encode() if body else None, headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=600))

def ask(messages, model, temperature=0.7, max_tokens=1500):
    out = http('/chat/completions', dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, chat_template_kwargs={'enable_thinking': False}))
    text = out['choices'][0]['message']['content'] or ''
    return re.sub('<think>.*?</think>', '', text, flags=re.S)

def failing_checks(feedback):
    labels = re.findall('^(\\w+): requirement:', feedback, re.M)
    return ', '.join(labels) if labels else feedback.splitlines()[0][:70]

def run_once(model, spec, rounds, tag, exp, show_code, explain, temperature, problem='lfsr', *, taxonomy_path=None, knowledge_path=None, category_overrides=(), exclude_categories=(), rules_limit=2, show_detect_text=False, rule_bytes=1200, context_tokens=8000, max_tokens=1500, log_sink=None, output_dir=None, **grade_options):
    history, records = ([], [])
    data = _problem(problem)
    name = data['module']
    prompt_context = prompt_problem(problem, spec=spec, explain=explain, show_code=show_code, taxonomy_path=taxonomy_path, knowledge_path=knowledge_path, category_overrides=category_overrides, exclude_categories=exclude_categories, rules_limit=rules_limit, show_detect_text=show_detect_text, rule_bytes=rule_bytes, context_tokens=context_tokens, max_tokens=max_tokens, **grade_options)
    outdir = pathlib.Path(output_dir or 'candidates/gen') / exp / name
    outdir.mkdir(parents=True, exist_ok=True)
    checker_options = feedback_options(prompt_context, grade_options)
    for rnd in range(rounds):
        messages = build_prompt(prompt_context, history)
        audit = prompt_audit(prompt_context)
        applicable, signatures, selected = (audit['categories'], audit['signatures'], audit['selected_rules'])
        taxonomy_digest, knowledge_digest = (audit['taxonomy_sha256'], audit['knowledge_sha256'])
        taxonomy_path, knowledge_path = (audit['taxonomy_path'], audit['knowledge_path'])
        input_budget = audit['prompt_byte_budget']
        if log_sink:
            log_sink(dict(event='request', run=tag, round=rnd, model=model, messages=messages, categories=applicable, signatures=signatures, selected_rules=selected, taxonomy_sha256=taxonomy_digest, knowledge_sha256=knowledge_digest, prompt_byte_budget=input_budget))
        start = time.monotonic()
        reply = ask(messages, model, temperature, max_tokens=max_tokens)
        elapsed = time.monotonic() - start
        code = extract(reply)
        if log_sink:
            log_sink(dict(event='reply', run=tag, round=rnd, reply=reply, code=code, elapsed_seconds=elapsed))
        candidate, verdict, diagnostics = (None, None, {})
        if code is None:
            code = ''
            score, feedback = no_module_result(diagnostics)
        else:
            path = outdir / f'{name}_{tag}_{rnd}.v'
            path.write_text(code)
            candidate = str(path)
            score, feedback = grade(problem, path, diagnostics=diagnostics, **checker_options)
            verdict = True if score == 1.0 and data.get('official') else official(problem, path)
        same = bool(history) and code.strip() == history[-1]['code'].strip()
        record = dict(event='round', version=AGENT_VERSION, exp=exp, run=tag, round=rnd, score=score, feedback=feedback, code=code, reply=reply, same_as_previous=same, problem=name, model=model, official_pass=verdict, candidate=candidate, messages=messages, elapsed_seconds=elapsed, temperature=temperature, max_tokens=max_tokens, thinking=False, context_tokens=context_tokens, prompt_utf8_bytes=prompt_size(messages), prompt_byte_budget=input_budget, categories=applicable, retrieval_signatures=signatures, selected_rule_keys=[entry['key'] for entry in selected if entry['key']], kb_hits=[entry['key'] for entry in selected if entry['key']], selected_rules=selected, diagnostics=diagnostics, checker_options=grade_options, spec_path=data.get('spec'), reference_path=data.get('ref_verilog'), seed=data.get('seed'), comparison_contracts=data.get('comparisons', {}), taxonomy_path=str(taxonomy_path), taxonomy_sha256=taxonomy_digest, knowledge_path=str(knowledge_path), knowledge_sha256=knowledge_digest, excluded_categories=list(exclude_categories), checker_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
        records.append(record)
        if log_sink:
            log_sink(record)
        history.append(record)
        print(f"run {tag} round {rnd}: {score:.2f}; checks={failing_checks(feedback)}; rules={record['selected_rule_keys']}; input_bytes={record['prompt_utf8_bytes']}/{input_budget}", flush=True)
        print(feedback, flush=True)
        if score == 1.0:
            break
    return records

def _agent_main(argv=None):
    global MODEL_BASE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--problem', help='convenience name for a same-stem Markdown specification')
    ap.add_argument('--spec', help='Markdown file; mine/bench remain aliases for the default problem')
    ap.add_argument('--tb', help='optional official testbench; same-stem testbenches are discovered')
    ap.add_argument('--ref', help='independent reference; same-stem references are discovered')
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--repeat', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cycles', type=int, default=256)
    ap.add_argument('--random-cycles', type=int, default=128)
    ap.add_argument('--facts', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--detectors', action='store_true')
    ap.add_argument('--principles', action='store_true', help='retrieve taxonomy rules for current failures')
    ap.add_argument('--principles-v2', action='store_true', help='enable pattern detectors and dynamic taxonomy retrieval')
    ap.add_argument('--explain', action='store_true')
    ap.add_argument('--hint', action='store_true')
    ap.add_argument('--no-code', action='store_true')
    ap.add_argument('--interface-from-tb', action='store_true')
    reset = ap.add_mutually_exclusive_group()
    reset.add_argument('--async-reset', dest='async_reset', action='store_true')
    reset.add_argument('--no-async-reset', dest='async_reset', action='store_false')
    ap.set_defaults(async_reset=None)
    ap.add_argument('--taxonomy', default=str(ROOT / 'taxonomy.json'))
    ap.add_argument('--knowledge', default=str(ROOT / 'knowledge.json'))
    ap.add_argument('--show-detect-text', action='store_true')
    ap.add_argument('--exclude-categories', default='', help='comma-separated categories excluded from retrieval')
    ap.add_argument('--category', action='append', default=[], help='override or supplement inferred categories')
    ap.add_argument('--rules-limit', type=int, default=2)
    ap.add_argument('--rule-bytes', type=int, default=1200)
    ap.add_argument('--context-tokens', type=int, default=8000)
    ap.add_argument('--max-tokens', type=int, default=1500)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--model')
    ap.add_argument('--base-url')
    ap.add_argument('--log-dir', default=str(ROOT / 'log'))
    ap.add_argument('--no-console-log', action='store_true')
    ap.add_argument('--run-name', default=AGENT_VERSION)
    ap.add_argument('--output-dir', default='candidates/gen')
    a = ap.parse_args(argv)
    if a.rounds < 1 or a.repeat < 0 or a.rules_limit < 1 or (a.cycles < 1) or (a.random_cycles < 1):
        ap.error('rounds, rules-limit, and cycle counts must be positive; repeat must be nonnegative')
    if a.problem and a.spec and (a.spec not in ('mine', 'bench')):
        ap.error('use --problem or --spec FILE')
    style = a.spec if a.spec in ('mine', 'bench') else 'mine'
    spec_file = _find_spec(a.problem or 'lfsr') if not a.spec or a.spec in ('mine', 'bench') else pathlib.Path(a.spec)
    try:
        problem = load_problem(spec_file, a.tb, a.ref, seed=a.seed, async_reset=a.async_reset, cycles=a.cycles, random_cycles=a.random_cycles)
        validate_knowledge(a.taxonomy, a.knowledge)
        spec = public_spec(problem, hint=a.hint or a.async_reset is True, interface_from_tb=a.interface_from_tb)
    except (OSError, ValueError) as error:
        ap.error(str(error))
    a.problem = problem['module']
    a.principles = a.principles or a.principles_v2
    a.detectors = a.detectors or a.principles_v2
    grade_options = dict(facts=a.facts, detectors=a.detectors, principles=a.principles, async_reset=a.async_reset)
    exp = style + ('_hint' if a.hint else '') + ('_nocode' if a.no_code else '') + ('_explain' if a.explain else '') + ('_principles' if a.principles else '') + ('_v2' if a.principles_v2 else '') + (f'_t{a.temperature}' if a.temperature != 0.7 else '')
    if a.base_url:
        MODEL_BASE = a.base_url.rstrip('/')
        MODEL_BASE = MODEL_BASE if MODEL_BASE.endswith('/v1') else MODEL_BASE + '/v1'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + str(time.time_ns())
    run_id = re.sub('[^\\w.-]+', '_', a.run_name) + '_' + a.problem + '_' + stamp
    directory = pathlib.Path(a.log_dir) / 'runs' / run_id
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / 'events.jsonl'
    solved, final_scores = (0, [])
    with log_path.open('w') as logfile:

        def write(record):
            logfile.write(json.dumps(record) + '\n')
            logfile.flush()
            if record.get('event') == 'request':
                save_prompt_event(record, log_path)
        logged_arguments = dict(vars(a))
        if not a.interface_from_tb:
            logged_arguments.pop('interface_from_tb')
        if not a.show_detect_text:
            logged_arguments.pop('show_detect_text')
        if not a.no_console_log:
            logged_arguments.pop('no_console_log')
        write(dict(event='start', version=AGENT_VERSION, run_id=run_id, arguments=logged_arguments, spec_path=str(spec_file), problem=problem['module'], base_url=MODEL_BASE))
        try:
            model = a.model or http('/models')['data'][0]['id']
            for run in range(a.repeat):
                records = run_once(model, spec, a.rounds, run, run_id, not a.no_code, a.explain, a.temperature, problem, taxonomy_path=a.taxonomy, knowledge_path=a.knowledge, category_overrides=a.category, exclude_categories=[value.strip() for value in a.exclude_categories.split(',') if value.strip()], rules_limit=a.rules_limit, show_detect_text=a.show_detect_text, rule_bytes=a.rule_bytes, context_tokens=a.context_tokens, max_tokens=a.max_tokens, log_sink=write, output_dir=a.output_dir, **grade_options)
                final_scores.append(records[-1]['score'])
                solved += records[-1]['score'] == 1.0
            write(dict(event='summary', solved=solved, repeat=a.repeat, final_scores=final_scores))
        except Exception as error:
            write(dict(event='error', type=type(error).__name__, message=str(error)))
            raise
    print(f'SOLVED {solved}/{a.repeat}; final_scores={final_scores}; log={log_path}', flush=True)
    return dict(a=a, exp=exp, spec=spec, problem=problem, grade_options=grade_options)

EMBEDDED_TESTS = {'test_checker.py': '"""Detector and grading regressions; also run by ``python checker.py``."""\n'
                    'import contextlib\n'
                    'import io\n'
                    'import json\n'
                    'import pathlib\n'
                    'import runpy\n'
                    'import subprocess\n'
                    'import sys\n'
                    'import tempfile\n'
                    'import unittest\n'
                    'from unittest import mock\n'
                    'import checker\n'
                    'ROOT = pathlib.Path(checker.__file__).resolve().parent\n'
                    "TB = ROOT / 'TestBench/lfsr_tb.v'\n"
                    "REFERENCE = ROOT / 'candidates/lfsr.v'\n"
                    '\n'
                    'class CheckerTests(unittest.TestCase):\n'
                    '\n'
                    '    def setUp(self):\n'
                    '        directory = '
                    "self.enterContext(tempfile.TemporaryDirectory(prefix='checker_test_'))\n"
                    '        self.directory = pathlib.Path(directory)\n'
                    "        self.enterContext(mock.patch.object(checker, 'BUILD', self.directory))\n"
                    '        self.enable_principles, self.enable_detectors = (False, True)\n'
                    "        entries = json.loads((ROOT / 'principles.json').read_text())\n"
                    "        self.principles = {entry['id']: 'Principle: ' + entry['principle'] for entry in "
                    'entries}\n'
                    '\n'
                    '    def grade(self, *args, **kwargs):\n'
                    '        return checker.grade(*args, principles=self.enable_principles or '
                    'self.enable_detectors, detectors=self.enable_detectors, **kwargs)\n'
                    '\n'
                    '    def principle_lines_for(self, *args):\n'
                    '        return checker._principle_lines(*args, principles=self.enable_principles or '
                    'self.enable_detectors, detectors=self.enable_detectors)\n'
                    '\n'
                    '    def patterns(self, feedback):\n'
                    "        return [line for line in feedback.splitlines() if line.startswith('Observed "
                    "pattern: ')]\n"
                    '\n'
                    '    def principle_lines(self, feedback):\n'
                    "        return [line for line in feedback.splitlines() if line.startswith('Principle: "
                    "')]\n"
                    '\n'
                    '    def design(self, name, update):\n'
                    "        path = self.directory / (name + '.v')\n"
                    '        path.write_text("module lfsr(input clk, input reset_n, output reg [7:0] '
                    'data);\\n  always @(posedge clk or negedge reset_n) begin\\n    if (!reset_n) data <= '
                    '8\'b10001010;\\n    else begin\\n      " + update + \'\\n    end\\n  '
                    "end\\nendmodule\\n')\n"
                    '        return path\n'
                    '\n'
                    '    def assert_modes(self, path, expected_score, signatures):\n'
                    '        baseline = None\n'
                    '        for v1, v2 in [(False, False), (True, False), (False, True), (True, True)]:\n'
                    '            with self.subTest(path=path.name, v1=v1, v2=v2):\n'
                    '                self.enable_principles, self.enable_detectors = (v1, v2)\n'
                    '                score, feedback = self.grade(TB, path)\n'
                    '                self.assertEqual(score, expected_score)\n'
                    '                if not v1 and (not v2):\n'
                    '                    baseline = feedback\n'
                    "                plain = '\\n'.join((line for line in feedback.splitlines() if not "
                    "line.startswith(('Principle: ', 'Observed pattern: '))))\n"
                    '                self.assertEqual(plain, baseline)\n'
                    '                self.assertEqual(self.patterns(feedback), [checker._v2_observation(s, '
                    '8) for s in signatures] if v2 else [])\n'
                    '                principles = self.principle_lines(feedback)\n'
                    '                self.assertLessEqual(len(principles), 2)\n'
                    '                if signatures and (v1 or v2):\n'
                    "                    v1_id = 'NEXT_STATE_WRONG_FIRST_EDGE' if 'first_step:' in feedback "
                    "else 'NEXT_STATE_DIVERGES_LATER'\n"
                    '                    expected = [self.principles[v1_id]]\n'
                    '                    if v2:\n'
                    '                        expected.append(self.principles[signatures[0]])\n'
                    '                    self.assertEqual(principles, expected)\n'
                    '                elif not v1 and (not v2):\n'
                    '                    self.assertEqual(principles, [])\n'
                    '                if score == 1.0:\n'
                    "                    self.assertEqual(feedback, 'correct')\n"
                    '\n'
                    '    def test_reference_and_existing_bad_designs(self):\n'
                    '        self.assert_modes(REFERENCE, 1.0, [])\n'
                    "        cases = {'lfsr_badtap.v': (0.55, ['SHIFTED_TOWARD_MSB', 'OFF_BY_ONE_VALUE']), "
                    "'lfsr_initial_hack.v': (0.8, []), 'lfsr_syncreset.v': (0.65, []), 'lfsr_two_drivers.v': "
                    "(0.65, []), 'shift_wrong_direction.v': (0.4, ['SHIFTED_TOWARD_LSB']), "
                    "'inverted_feedback.v': (0.4, ['SHIFTED_TOWARD_MSB', 'OFF_BY_ONE_VALUE'])}\n"
                    '        for name, (score, signatures) in cases.items():\n'
                    "            self.assert_modes(ROOT / 'candidates/bad' / name, score, signatures)\n"
                    '\n'
                    '    def test_every_detector_on_bad_designs(self):\n'
                    '        cases = {\'partial_x\': ("8\'bx0010101", [\'PARTIAL_X\']), \'stuck_and_late\': '
                    "('data', ['OUTPUT_STUCK', 'ONE_CYCLE_LATE']), 'inverted_output': "
                    '("~8\'b00010101", [\'OUTPUT_INVERTED\']), \'reversed_output\': ("8\'b10101000", '
                    '[\'BIT_ORDER_REVERSED\']), \'shift_lsb\': ("{1\'b0, data[7:1]}", '
                    '[\'SHIFTED_TOWARD_LSB\']), \'shift_msb\': ("{data[6:0], 1\'b0}", '
                    '[\'SHIFTED_TOWARD_MSB\', \'OFF_BY_ONE_VALUE\']), \'off_by_one\': ("8\'b00010110", '
                    '[\'OFF_BY_ONE_VALUE\']), \'early\': ("8\'b00101011", [\'ONE_CYCLE_EARLY\']), '
                    '\'upper_zero\': ("8\'b00000101", [\'UPPER_BITS_ZERO\'])}\n'
                    '        covered = set()\n'
                    '        for name, (value, signatures) in cases.items():\n'
                    "            path = self.design(name, f'data <= {value};')\n"
                    '            self.assert_modes(path, 0.4, signatures)\n'
                    '            _, feedback = self.grade(TB, path)\n'
                    "            self.assertNotIn('00010101', feedback)\n"
                    '            covered.update(signatures)\n'
                    '        self.assertEqual(covered, set(checker.V2_OBSERVATIONS))\n'
                    '\n'
                    '    def test_guards_and_reference_samples(self):\n'
                    '        detect = checker._v2_signatures\n'
                    "        for observed in ['', '10101', '000010101', 'xxxxxxxx', 'zzzzzzzz', 'xxxxxxxx!', "
                    "'00010?01']:\n"
                    '            with self.subTest(observed=observed):\n'
                    "                self.assertEqual(detect(observed, 21, '10001010', 8, 138, 43), [])\n"
                    "        for observed in ['X0010101', '0001010Z', 'xz01xz01']:\n"
                    "            self.assertEqual(detect(observed, 21, '10001010', 8, 138, 43), "
                    "['PARTIAL_X'])\n"
                    "        for previous in ['', 'xxxxxxxx', '1000101x', '1010']:\n"
                    "            self.assertEqual(detect('00010100', 21, previous, 8, 138, 43), "
                    "['OFF_BY_ONE_VALUE'])\n"
                    "        state = checker.PROBLEMS['lfsr']['model']['reset_state']\n"
                    '        for _ in range(256):\n'
                    '            expected = checker._lfsr_step(state)\n'
                    "            self.assertEqual(detect(format(expected, '08b'), expected, format(state, "
                    "'08b'), 8, state, checker._lfsr_step(expected)), [])\n"
                    '            state = expected\n'
                    "        self.assertNotIn('OFF_BY_ONE_VALUE', detect('11111111', 0, '10001010', 8, 138, "
                    '1))\n'
                    "        self.assertNotIn('OFF_BY_ONE_VALUE', detect('00000000', 255, '10001010', 8, "
                    '138, 1))\n'
                    "        self.assertNotIn('UPPER_BITS_ZERO', detect('00000100', 21, '10001010', 8, 138, "
                    '43))\n'
                    "        self.assertNotIn('UPPER_BITS_ZERO', detect('00000101', 6, '10001010', 8, 138, "
                    '43))\n'
                    '\n'
                    '    def test_observed_previous_state_and_entered_bit(self):\n'
                    '        detect = checker._v2_signatures\n'
                    "        for observed in ['01000101', '11000101']:\n"
                    "            self.assertEqual(detect(observed, 21, '10001010', 8, 18, 43), "
                    "['SHIFTED_TOWARD_LSB'])\n"
                    "        for observed in ['00010100', '00010101']:\n"
                    "            self.assertEqual(detect(observed, 119, '10001010', 8, 18, 43), "
                    "['SHIFTED_TOWARD_MSB'])\n"
                    "        self.assertEqual(detect('10001010', 21, '10001010', 8, 18, 43), "
                    "['OUTPUT_STUCK'])\n"
                    "        self.assertEqual(detect('00010010', 21, '10001010', 8, 18, 43), "
                    "['ONE_CYCLE_LATE'])\n"
                    '\n'
                    '    def test_only_first_failing_cycle_is_reported(self):\n'
                    '        path = self.design(\'changing_patterns\', "if (data == 8\'h8a) data <= '
                    "8'h15;\\n      else if (data == 8'h15) data <= 8'h2a;\\n      else data <= "
                    '8\'bxxxxxxxx;")\n'
                    '        _, feedback = self.grade(TB, path)\n'
                    "        self.assertIn('at cycle 2 after release', feedback)\n"
                    "        self.assertNotIn('first_step:', feedback)\n"
                    '        self.assertEqual(self.patterns(feedback), [checker._v2_observation(s, 8) for s '
                    "in ['SHIFTED_TOWARD_MSB', 'OFF_BY_ONE_VALUE']])\n"
                    '\n'
                    '    def test_detector_is_never_called_when_disabled(self):\n'
                    '        self.enable_detectors = False\n'
                    "        self.assertEqual(checker._v2_signatures('00010100', 21, '10001010', 8, 138, 43, "
                    'detectors=False), [])\n'
                    "        with mock.patch.object(checker, '_v2_signatures', "
                    "side_effect=AssertionError('v2 was called')):\n"
                    '            for v1 in [False, True]:\n'
                    '                self.enable_principles = v1\n'
                    '                score, feedback = self.grade(TB, ROOT / '
                    "'candidates/bad/inverted_feedback.v')\n"
                    '                self.assertEqual(score, 0.4)\n'
                    '                self.assertEqual(self.patterns(feedback), [])\n'
                    '\n'
                    '    def test_principle_priority_budget_and_deduplication(self):\n'
                    "        v1 = ['X_BEFORE_FIRST_EDGE', 'MULTIPLE_DRIVERS']\n"
                    "        v2 = ['PARTIAL_X', 'OUTPUT_STUCK', 'ONE_CYCLE_LATE']\n"
                    '        self.assertEqual(self.principle_lines_for(v1, v2), [self.principles[s] for s in '
                    'v1])\n'
                    '        self.assertEqual(self.principle_lines_for(v1[:1] * 2, v2), '
                    '[self.principles[v1[0]], self.principles[v2[0]]])\n'
                    '        self.assertEqual(self.principle_lines_for([], v2), [self.principles[s] for s in '
                    'v2[:2]])\n'
                    '        self.enable_principles, self.enable_detectors = (True, False)\n'
                    '        self.assertEqual(self.principle_lines_for(v1[:1], v2), '
                    '[self.principles[v1[0]]])\n'
                    '        self.enable_principles = False\n'
                    '        self.assertEqual(self.principle_lines_for(v1, v2), [])\n'
                    '\n'
                    '    def test_compile_failure_and_timeout(self):\n'
                    "        failed = subprocess.CompletedProcess([], 1, stdout='', stderr='bad syntax')\n"
                    "        compiled = subprocess.CompletedProcess([], 0, stdout='', stderr='')\n"
                    "        cases = [([failed], 0.0, 'COMPILE_ERROR'), ([compiled, "
                    "subprocess.TimeoutExpired('vvp', 1)], 0.1, 'TIMEOUT')]\n"
                    '        for responses, score, signature in cases:\n'
                    "            with mock.patch.object(checker.subprocess, 'run', side_effect=responses):\n"
                    '                result, feedback = self.grade(TB, REFERENCE)\n'
                    '            self.assertEqual(result, score)\n'
                    '            self.assertEqual(self.patterns(feedback), [])\n'
                    '            self.assertEqual(self.principle_lines(feedback), '
                    '[self.principles[signature]])\n'
                    '\n'
                    '    def test_cli_flags_and_experiment_name(self):\n'
                    "        cases = [([], 'mine', False, False), (['--principles'], 'mine_principles', "
                    "True, False), (['--principles-v2'], 'mine_principles_v2', True, True)]\n"
                    '        for flags, experiment, v1, v2 in cases:\n'
                    '            response = io.BytesIO(b\'{"data": [{"id": "test-model"}]}\')\n'
                    '            with contextlib.chdir(self.directory), '
                    "contextlib.redirect_stdout(io.StringIO()), mock.patch.object(sys, 'argv', ['agent.py', "
                    "'--repeat', '0'] + flags), mock.patch('urllib.request.urlopen', "
                    'return_value=response):\n'
                    '                result = checker._agent_main(sys.argv[1:])\n'
                    "            self.assertEqual(result['exp'], experiment)\n"
                    "            self.assertEqual(result['a'].principles, v1)\n"
                    "            self.assertEqual(result['a'].detectors, v2)\n"
                    "if __name__ == '__main__':\n"
                    '    unittest.main()',
 'test_engine.py': '"""Shared-engine, sequence behavior A, and agent integration regressions."""\n'
                   'import contextlib\n'
                   'import io\n'
                   'import pathlib\n'
                   'import re\n'
                   'import runpy\n'
                   'import subprocess\n'
                   'import sys\n'
                   'import tempfile\n'
                   'import unittest\n'
                   'from unittest import mock\n'
                   'import agent\n'
                   'import checker\n'
                   'ROOT = pathlib.Path(checker.__file__).resolve().parent\n'
                   "SEQ_REFERENCE = ROOT / 'reference/seq/seq_a.v'\n"
                   "SEQ_BAD = ROOT / 'candidates/bad/sequence_generator'\n"
                   '\n'
                   'class EngineTests(unittest.TestCase):\n'
                   '\n'
                   '    def setUp(self):\n'
                   '        self.directory = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))\n'
                   "        self.enterContext(mock.patch.object(checker, 'BUILD', self.directory))\n"
                   '        self.enable_principles = self.enable_detectors = False\n'
                   '\n'
                   '    def grade(self, *args, **kwargs):\n'
                   '        return checker.grade(*args, principles=self.enable_principles or '
                   'self.enable_detectors, detectors=self.enable_detectors, **kwargs)\n'
                   '\n'
                   '    def design(self, name, source):\n'
                   "        path = self.directory / (name + '.v')\n"
                   '        path.write_text(source)\n'
                   '        return path\n'
                   '\n'
                   '    def test_problem_weights_and_reference_models(self):\n'
                   '        for name, p in checker.PROBLEMS.items():\n'
                   '            with self.subTest(problem=name):\n'
                   "                self.assertAlmostEqual(sum(p['weights'].values()), 1.0)\n"
                   "                outputs = {port for port, direction, _ in p['ports'] if direction == "
                   "'output'}\n"
                   "                self.assertEqual(set(p['model']['output'](p['model']['reset_state'])), "
                   'outputs)\n'
                   "                for segment in p['segments']:\n"
                   "                    self.assertIn(segment['label'], p['weights'])\n"
                   "                    if isinstance(segment['inputs'], list):\n"
                   "                        self.assertEqual(len(segment['inputs']), segment['cycles'])\n"
                   '\n'
                   '    def test_sequence_reference_and_official(self):\n'
                   "        self.assertEqual(self.grade('sequence_generator', SEQ_REFERENCE), (1.0, "
                   "'correct'))\n"
                   "        self.assertIs(checker.official('sequence_generator', SEQ_REFERENCE), True)\n"
                   "        score, feedback = self.grade('sequence_generator', ROOT / "
                   "'reference/seq/seq_b.v')\n"
                   '        self.assertLess(score, 1.0)\n'
                   "        self.assertIn('first_step: requirement:', feedback)\n"
                   '\n'
                   '    def test_every_legacy_sequence_fault_and_feedback_modes(self):\n'
                   "        checks = {'no_wrap': 'repetition', 'ignores_enable': 'disable', "
                   "'reset_position': 'reset_restart', 'bad_item': 'sequence', 'restart_on_pause': "
                   "'resume'}\n"
                   '        for name, required in checks.items():\n'
                   '            baseline = None\n'
                   '            for v1, v2 in ((False, False), (True, False), (False, True), (True, True)):\n'
                   '                with self.subTest(design=name, v1=v1, v2=v2):\n'
                   '                    self.enable_principles, self.enable_detectors = (v1, v2)\n'
                   "                    score, feedback = self.grade('sequence_generator', SEQ_BAD / (name + "
                   "'.v'))\n"
                   '                    self.assertLess(score, 1.0)\n'
                   "                    self.assertIn(required + ': requirement:', feedback)\n"
                   "                    labels = re.findall('^(\\\\w+): requirement:', feedback, re.M)\n"
                   '                    self.assertEqual(len(labels), len(set(labels)))\n'
                   "                    plain = '\\n'.join((line for line in feedback.splitlines() if not "
                   "line.startswith(('Principle:', 'Observed pattern:'))))\n"
                   '                    if baseline is None:\n'
                   '                        baseline = (score, plain)\n'
                   '                    self.assertEqual((score, plain), baseline)\n'
                   '                    principles = [line for line in feedback.splitlines() if '
                   "line.startswith('Principle:')]\n"
                   '                    self.assertLessEqual(len(principles), 2)\n'
                   '                    if v1 or v2:\n'
                   '                        self.assertTrue(principles)\n'
                   "                    if v2 and name != 'ignores_enable':\n"
                   "                        self.assertIn('Observed pattern:', feedback)\n"
                   '                    elif not v2:\n'
                   "                        self.assertNotIn('Observed pattern:', feedback)\n"
                   '                    self.assertNotRegex(feedback, '
                   "'(?i)expected|8\\\\x27[hbd]|0x[0-9a-f]|\\\\b[01]{8}\\\\b')\n"
                   '\n'
                   '    def test_reset_while_enabled_and_disabled_independently(self):\n'
                   '        source = SEQ_REFERENCE.read_text()\n'
                   '        for enabled in (0, 1):\n'
                   '            with self.subTest(enabled=enabled):\n'
                   "                text = source.replace('  reg [2:0] idx;', '  reg [2:0] idx; reg started "
                   "= 0;')\n"
                   '                text = text.replace("idx <= 3\'d1; data <= 8\'hAF;", f"idx <= (enable == '
                   '{enabled} && started) ? 3\'d0 : 3\'d1; data <= 8\'hAF;")\n'
                   "                text = text.replace('else if (enable) begin', 'else if (enable) begin "
                   "started <= 1;')\n"
                   "                path = self.design('wrong_reset_' + str(enabled), text)\n"
                   "                _, feedback = self.grade('sequence_generator', path)\n"
                   "                self.assertIn('reset_restart: requirement:', feedback)\n"
                   "                self.assertNotIn('reset_value:', feedback)\n"
                   "                self.assertNotIn('first_step:', feedback)\n"
                   '\n'
                   '    def test_reset_checks_and_facts_apply_to_sequence(self):\n'
                   '        source = SEQ_REFERENCE.read_text()\n'
                   "        variants = {'sync': source.replace(' or negedge reset_n', ''), "
                   '\'wrong_reset_value\': source.replace("data <= 8\'hAF; end", "data <= 8\'h00; end")}\n'
                   '        self.enable_principles = True\n'
                   '        for name, text in variants.items():\n'
                   '            with self.subTest(name=name):\n'
                   '                path = self.design(name, text)\n'
                   "                _, feedback = self.grade('sequence_generator', path)\n"
                   "                self.assertIn('reset_immediate: requirement:', feedback)\n"
                   "                self.assertIn('reset_midrun: requirement:', feedback)\n"
                   "                self.assertIn('In your design, data is assigned in 1 place(s):', "
                   'feedback)\n'
                   "                self.assertIn('Principle:', feedback)\n"
                   '\n'
                   '    def test_sequence_plan_covers_lengths_and_reset_positions(self):\n'
                   "        p = checker.PROBLEMS['sequence_generator']\n"
                   '        _, samples = checker._probe(p)\n'
                   "        for label, count in [('sequence', 8), ('repetition', 16), ('disable', 5), "
                   "('resume', 16)]:\n"
                   "            self.assertEqual(sum((label in s['checks'] and s['kind'] == 'segment' for s "
                   'in samples)), count)\n'
                   '        for i, sample in enumerate(samples):\n'
                   "            if sample['kind'] == 'reset' and sample['checks'] == ('reset_restart',) and "
                   "(sample['subcheck'] == 0):\n"
                   "                self.assertNotEqual(samples[i - 1]['want']['data'], "
                   'checker.SEQUENCE[0])\n'
                   "        initial_output = p['model']['output'](p['model']['reset_state'])['data']\n"
                   '        self.assertEqual(initial_output, checker.SEQUENCE[0])\n'
                   "        first = next((s for s in samples if 'first_step' in s['checks']))\n"
                   "        self.assertEqual(first['want']['data'], checker.SEQUENCE[1])\n"
                   '\n'
                   '    def synthetic_problem(self):\n'
                   '        """Third problem proves ports, polarity, reset type and inputs are data."""\n'
                   "        return dict(module='counter', ports=(('tick', 'input', 1), ('clear', 'input', "
                   "1), ('step', 'input', 2), ('count', 'output', 3), ('odd', 'output', 1)), clock='tick', "
                   "reset='clear', reset_active=1, async_reset=False, model=dict(reset_state=0, "
                   "next_state=lambda s, i: (s + i['step']) % 8, output=lambda s: {'count': s, 'odd': s & "
                   "1}), initial_inputs={'step': 0}, reset_hold_cycles=2, restart_cycles=4, "
                   "segments=[dict(label='sequence', cycles=5, inputs=[{'step': n} for n in (1, 2, 0, 3, "
                   "1)], first_check='first_step', requirement='each rising edge must add step to count')], "
                   "midrun_inputs={'step': 1}, requirements=dict(checker.RESET_REQUIREMENTS, "
                   "reset_midrun='reset held across a clock edge must restore the initial state and restart "
                   "after release', first_step='the first rising edge must add step'), observations={}, "
                   'weights=dict(compiles=0.1, reset_value=0.15, reset_midrun=0.2, first_step=0.15, '
                   "sequence=0.4), official=str(self.directory / 'no_official_testbench.v'))\n"
                   '\n'
                   '    def test_multi_output_sync_active_high_and_settled_sampling(self):\n'
                   '        p = self.synthetic_problem()\n'
                   "        source = '`timescale 1ns/1ps\\nmodule counter(input tick, clear, input [1:0] "
                   'step, output reg [2:0] count, output reg odd);\\nalways @(posedge tick) begin\\n  if '
                   '(clear) begin count <= #0.8 0; odd <= #0.8 0; end\\n  else begin count <= #0.8 count + '
                   'step; odd <= #0.8 ((count + step) & 1); end\\nend\\n// Catch any probe that changes '
                   'stimulus at a rising edge or while clock is high.\\nalways @(step or clear) if (tick) '
                   '$fatal(1, "stimulus changed with clock high");\\nendmodule\\n\'\n'
                   "        path = self.design('counter', source)\n"
                   "        with mock.patch.dict(checker.PROBLEMS, {'counter': p}):\n"
                   "            self.assertEqual(self.grade('counter', path), (1.0, 'correct'))\n"
                   '            harness, samples = checker._probe(p)\n'
                   "            self.assertFalse(any((s['kind'] == 'initial_immediate' for s in samples)))\n"
                   "            self.assertFalse(any((s['kind'] == 'reset' and s['subcheck'] == 0 for s in "
                   'samples)))\n'
                   "            self.assertNotIn('reset_immediate', harness)\n"
                   '            self.enable_detectors = True\n'
                   "            wrong = self.design('wrong_odd', source.replace('((count + step) & 1)', "
                   "'0'))\n"
                   "            score, feedback = self.grade('counter', wrong)\n"
                   '            self.assertLess(score, 1.0)\n'
                   "            self.assertIn('odd is wrong in bit(s) [0]', feedback)\n"
                   "            self.assertNotIn('reset_immediate:', feedback)\n"
                   "            self.assertIn('Observed pattern:', feedback)\n"
                   "            self.assertIn('Principle:', feedback)\n"
                   '\n'
                   '    def test_early_finish_and_official_failure_do_not_pass(self):\n'
                   "        path = self.design('early_finish', SEQ_REFERENCE.read_text().replace('  reg "
                   "[2:0] idx;', '  reg [2:0] idx; initial #12 $finish;'))\n"
                   "        score, feedback = self.grade('sequence_generator', path)\n"
                   '        self.assertLess(score, 1.0)\n'
                   "        self.assertIn('not observed (the simulation ended early)', feedback)\n"
                   "        with mock.patch.object(checker, '_official_ok', return_value=False):\n"
                   "            score, feedback = self.grade('sequence_generator', SEQ_REFERENCE)\n"
                   '        self.assertEqual(score, 0.9)\n'
                   "        self.assertIn('official: requirement:', feedback)\n"
                   '\n'
                   '    def test_sequence_compile_and_timeout_principles(self):\n'
                   '        self.enable_principles = True\n'
                   "        failed = subprocess.CompletedProcess([], 1, stdout='', stderr='bad syntax')\n"
                   "        compiled = subprocess.CompletedProcess([], 0, stdout='', stderr='')\n"
                   '        for responses, expected in [([failed], 0.0), ([compiled, '
                   "subprocess.TimeoutExpired('vvp', 1)], 0.1)]:\n"
                   "            with mock.patch.object(checker.subprocess, 'run', side_effect=responses):\n"
                   "                score, feedback = self.grade('sequence_generator', SEQ_REFERENCE)\n"
                   '            self.assertEqual(score, expected)\n'
                   "            self.assertIn('Principle:', feedback)\n"
                   '\n'
                   '    def test_agent_cli_problem_and_all_feedback_flags(self):\n'
                   '        for problem in checker.PROBLEMS:\n'
                   "            for flags, v1, v2 in [(['--explain'], False, False), (['--explain', "
                   "'--principles'], True, False), (['--explain', '--principles-v2'], True, True)]:\n"
                   '                with self.subTest(problem=problem, flags=flags):\n'
                   '                    response = io.BytesIO(b\'{"data": [{"id": "test-model"}]}\')\n'
                   '                    with contextlib.chdir(self.directory), '
                   "contextlib.redirect_stdout(io.StringIO()), mock.patch.object(sys, 'argv', ['agent.py', "
                   "'--repeat', '0', '--problem', problem] + flags), mock.patch('urllib.request.urlopen', "
                   'return_value=response):\n'
                   '                        result = checker._agent_main(sys.argv[1:])\n'
                   "                    self.assertEqual(result['a'].problem, problem)\n"
                   "                    self.assertTrue(result['a'].explain)\n"
                   "                    self.assertEqual((result['a'].principles, result['a'].detectors), "
                   '(v1, v2))\n'
                   "                    self.assertIn(problem, result['spec'])\n"
                   '\n'
                   '    def test_agent_sequence_rounds_use_shared_grader_and_diagnosis(self):\n'
                   '        self.enable_principles = self.enable_detectors = True\n'
                   '        replies = [f"```verilog\\n{(SEQ_BAD / \'bad_item.v\').read_text()}\\n```", '
                   "f'```verilog\\n{SEQ_REFERENCE.read_text()}\\n```']\n"
                   "        config = dict(checker.PROBLEMS['sequence_generator'], official=str(ROOT / "
                   "'TestBench/sequence_generator_tb.v'))\n"
                   '        with contextlib.chdir(self.directory), '
                   "contextlib.redirect_stdout(io.StringIO()), mock.patch.object(agent, 'ask', "
                   "side_effect=replies) as ask, mock.patch.dict(checker.PROBLEMS, {'sequence_generator': "
                   'config}):\n'
                   "            log = agent.run_once('test-model', "
                   "checker.problem_spec('sequence_generator'), 2, 0, 'test', True, True, 0.7, "
                   "'sequence_generator', principles=True, detectors=True)\n"
                   "        self.assertEqual([entry['score'] for entry in log], [0.4, 1.0])\n"
                   "        self.assertIs(log[-1]['official_pass'], True)\n"
                   "        self.assertEqual(log[-1]['problem'], 'sequence_generator')\n"
                   "        followup = ask.call_args_list[1].args[0][0]['content']\n"
                   "        self.assertIn('say in one sentence', followup)\n"
                   "        self.assertIn('Principle:', followup)\n"
                   "        self.assertIn('Observed pattern:', followup)\n"
                   "        for style in ('mine', 'bench'):\n"
                   "            spec = checker.problem_spec('sequence_generator', style, True)\n"
                   "            self.assertIn('first enabled edge produces BC', spec)\n"
                   "            self.assertIn('asynchronous', spec)\n"
                   '\n'
                   'class FileReferenceTests(unittest.TestCase):\n'
                   '\n'
                   '    def setUp(self):\n'
                   '        self.directory = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))\n'
                   "        self.enterContext(mock.patch.object(checker, 'BUILD', self.directory))\n"
                   "        self.problem = checker.file_problem(ROOT / 'sequence_generator.md', ROOT / "
                   "'TestBench/sequence_generator_tb.v', SEQ_REFERENCE)\n"
                   '\n'
                   '    def test_both_reference_sources_and_all_bad_designs(self):\n'
                   "        registered = dict(checker.PROBLEMS['sequence_generator'], "
                   'ref_verilog=str(SEQ_REFERENCE))\n'
                   "        registered['model'] = {'reset_state': None, 'next_state': lambda *_: "
                   "self.fail('Python model used'), 'output': lambda *_: self.fail('Python model used')}\n"
                   '        for config in (registered, self.problem):\n'
                   "            with self.subTest(config='registered' if config is registered else "
                   "'files'):\n"
                   '                self.assertEqual(checker.grade(config, SEQ_REFERENCE), (1.0, '
                   "'correct'))\n"
                   "                for path in sorted(SEQ_BAD.glob('*.v')):\n"
                   '                    score, feedback = checker.grade(config, path, detectors=True, '
                   'principles=True)\n'
                   '                    self.assertLess(score, 1.0)\n'
                   '                    self.assertNotRegex(feedback, '
                   "'(?i)expected|8\\\\x27[hbd]|0x[0-9a-f]|\\\\b[01]{8}\\\\b')\n"
                   "        score, feedback = checker.grade(self.problem, SEQ_BAD / 'ignores_enable.v')\n"
                   "        self.assertIn('1 cycle after enable went low', feedback)\n"
                   "        self.assertIn('data is wrong in bit(s)', feedback)\n"
                   '\n'
                   '    def test_flags_are_independent_and_do_not_mutate_problem_data(self):\n'
                   "        path = ROOT / 'candidates/bad/lfsr_syncreset.v'\n"
                   "        baseline = checker.grade('lfsr', path)\n"
                   "        _, without_facts = checker.grade('lfsr', path, facts=False)\n"
                   "        self.assertIn('In your design,', baseline[1])\n"
                   "        self.assertNotIn('In your design,', without_facts)\n"
                   "        with mock.patch.object(checker, '_official_ok', return_value=True):\n"
                   "            self.assertEqual(checker.grade('lfsr', path, async_reset=False), (1.0, "
                   "'correct'))\n"
                   "        self.assertEqual(checker.grade('lfsr', path), baseline)\n"
                   "        bad = SEQ_BAD / 'bad_item.v'\n"
                   "        _, detector_feedback = checker.grade('sequence_generator', bad, detectors=True)\n"
                   "        self.assertIn('Observed pattern:', detector_feedback)\n"
                   "        self.assertNotIn('Principle:', detector_feedback)\n"
                   "        _, principle_feedback = checker.grade('sequence_generator', bad, "
                   'principles=True)\n'
                   "        self.assertIn('Principle:', principle_feedback)\n"
                   "        self.assertNotIn('Observed pattern:', principle_feedback)\n"
                   '\n'
                   '    def test_seeded_probe_and_inferred_ports(self):\n'
                   "        self.assertEqual(self.problem['ports'], "
                   "checker.PROBLEMS['sequence_generator']['ports'])\n"
                   '        self.assertEqual(checker._harness(self.problem), '
                   "checker._harness(checker.file_problem(ROOT / 'sequence_generator.md', ROOT / "
                   "'TestBench/sequence_generator_tb.v', SEQ_REFERENCE)))\n"
                   "        other = checker.file_problem(ROOT / 'sequence_generator.md', ROOT / "
                   "'TestBench/sequence_generator_tb.v', SEQ_REFERENCE, seed=12)\n"
                   '        self.assertNotEqual(checker._harness(self.problem), checker._harness(other))\n'
                   '        for sample in checker._probe(self.problem)[1]:\n'
                   "            self.assertEqual(sample['want'], {})\n"
                   '\n'
                   '    def test_cli_file_only_needs_no_registry_entry(self):\n'
                   "        argv = ['checker.py', '--spec', str(ROOT / 'sequence_generator.md'), '--tb', "
                   "str(ROOT / 'TestBench/sequence_generator_tb.v'), '--ref', str(SEQ_REFERENCE), '--dut', "
                   "str(SEQ_REFERENCE), '--facts', '--detectors', '--principles', '--async-reset']\n"
                   '        with mock.patch.dict(checker.PROBLEMS, {}, clear=True), mock.patch.object(sys, '
                   "'argv', argv), contextlib.redirect_stdout(io.StringIO()) as output:\n"
                   '            self.assertEqual(checker.grade(self.problem, SEQ_REFERENCE), (1.0, '
                   "'correct'))\n"
                   '            with self.assertRaises(SystemExit) as exit:\n'
                   "                runpy.run_path(str(ROOT / 'checker.py'), run_name='__main__')\n"
                   '        self.assertEqual(exit.exception.code, 0)\n'
                   "        self.assertEqual(output.getvalue(), 'score: 1.0\\ncorrect\\n')\n"
                   '\n'
                   '    def test_agent_file_only_passes_all_options(self):\n'
                   '        response = io.BytesIO(b\'{"data": [{"id": "test-model"}]}\')\n'
                   "        argv = ['agent.py', '--spec', str(ROOT / 'sequence_generator.md'), '--tb', "
                   "str(ROOT / 'TestBench/sequence_generator_tb.v'), '--ref', str(SEQ_REFERENCE), "
                   "'--repeat', '0', '--explain', '--principles-v2', '--async-reset', '--no-facts']\n"
                   "        with contextlib.chdir(self.directory), mock.patch.object(sys, 'argv', argv), "
                   "mock.patch('urllib.request.urlopen', return_value=response), "
                   'contextlib.redirect_stdout(io.StringIO()):\n'
                   '            result = checker._agent_main(sys.argv[1:])\n'
                   "        self.assertEqual(result['problem']['ref_verilog'], str(SEQ_REFERENCE))\n"
                   "        self.assertEqual(result['grade_options'], dict(facts=False, detectors=True, "
                   'principles=True, async_reset=True))\n'
                   '        replies = [f"```verilog\\n{(SEQ_BAD / \'bad_item.v\').read_text()}\\n```", '
                   "f'```verilog\\n{SEQ_REFERENCE.read_text()}\\n```']\n"
                   '        with contextlib.chdir(self.directory), '
                   "contextlib.redirect_stdout(io.StringIO()), mock.patch.object(agent, 'ask', "
                   "side_effect=replies), mock.patch.object(checker, 'grade', wraps=checker.grade) as "
                   'grade:\n'
                   "            log = agent.run_once('test', result['spec'], 2, 0, 'files', True, True, 0.7, "
                   "result['problem'], **result['grade_options'])\n"
                   "        self.assertLess(log[0]['score'], 1.0)\n"
                   "        self.assertEqual(log[-1]['score'], 1.0)\n"
                   '        passed = grade.call_args.kwargs\n'
                   "        self.assertIn('diagnostics', passed)\n"
                   "        self.assertFalse(passed['principles'])\n"
                   "        for option in ('facts', 'detectors', 'async_reset'):\n"
                   "            self.assertEqual(passed[option], result['grade_options'][option])\n"
                   '\n'
                   '    def test_lfsr_stdout_is_byte_identical(self):\n'
                   "        result = subprocess.run([sys.executable, 'checker.py'], cwd=ROOT, "
                   'capture_output=True)\n'
                   '        self.assertEqual(result.returncode, 0, result.stderr.decode())\n'
                   "        self.assertEqual(result.stdout, (ROOT / 'baseline_lfsr.txt').read_bytes())\n"
                   "if __name__ == '__main__':\n"
                   '    unittest.main()',
 'test_taxonomy.py': '"""Classification, bounded principle retrieval, and read-only knowledge reports."""\n'
                     'import json\n'
                     'import contextlib\n'
                     'import io\n'
                     'import pathlib\n'
                     'import tempfile\n'
                     'import unittest\n'
                     'import checker\n'
                     'import taxonomy\n'
                     '\n'
                     'class TaxonomyTests(unittest.TestCase):\n'
                     '\n'
                     '    def test_keywords_match_whole_words_and_phrases(self):\n'
                     "        path = checker.ROOT / 'taxonomy.json'\n"
                     "        found = taxonomy.categories('data comes from the source when already set', "
                     'path=path)\n'
                     "        self.assertNotIn('memory', found)\n"
                     "        self.assertNotIn('handshake', found)\n"
                     "        self.assertIn('memory', taxonomy.categories('8-bit RAM with read enable', "
                     'path=path))\n'
                     "        definitions = {'categories': [{'name': 'test', 'spec_keywords': ['read "
                     "enable']}]}\n"
                     "        self.assertIn('test', taxonomy.categories('READ ENABLE asserted', "
                     'data=definitions))\n'
                     "        self.assertNotIn('test', taxonomy.categories('bread enabled', "
                     'data=definitions))\n'
                     '\n'
                     '    def test_only_principles_are_sent_by_default(self):\n'
                     "        entry = dict(signature='SIG', description='some detection', rule='Keep state "
                     "stable.')\n"
                     "        self.assertEqual(taxonomy.entry_text(entry), 'Principle: Keep state stable.')\n"
                     "        self.assertEqual(taxonomy.entry_text(entry, show_detect_text=True), 'Detected "
                     "signature SIG: some detection.\\nPrinciple: Keep state stable.')\n"
                     "        self.assertEqual(taxonomy.entry_text(dict(signature='SIG', description='some "
                     "detection')), '')\n"
                     '\n'
                     '    def test_missing_principles_do_not_take_slots_or_bytes(self):\n'
                     "        definitions = dict(signatures={'EMPTY': {'detect': 'A' * 1000}, 'GOOD': "
                     "{'detect': 'B' * 1000}})\n"
                     '        for verbose in (False, True):\n'
                     "            entries = taxonomy.select(None, ['generic'], ['EMPTY', 'GOOD'], limit=1, "
                     "max_bytes=2000, data=definitions, knowledge={'generic:GOOD': 'Keep state stable.'}, "
                     'show_detect_text=verbose)\n'
                     "            self.assertEqual([entry['key'] for entry in entries], ['generic:GOOD'])\n"
                     "            self.assertEqual(entries[0]['signature'], 'GOOD')\n"
                     "        entries = taxonomy.select(None, ['generic'], ['EMPTY', 'GOOD'], limit=1, "
                     "max_bytes=40, data=definitions, knowledge={'generic:GOOD': 'Keep state stable.'})\n"
                     '        self.assertEqual(len(entries), 1)\n'
                     "        self.assertEqual(taxonomy.select(None, ['generic'], ['GOOD'], limit=0, "
                     "data=definitions, knowledge={'generic:GOOD': 'Keep state stable.'}), [])\n"
                     '\n'
                     '    def test_empty_category_rule_can_fall_back_to_generic(self):\n'
                     "        entries = taxonomy.select(None, ['lfsr', 'generic'], ['SIG'], "
                     "data={'signatures': {}}, knowledge={'lfsr:SIG': '', 'generic:SIG': 'Hold state.'})\n"
                     "        self.assertEqual(entries[0]['key'], 'generic:SIG')\n"
                     '\n'
                     '    def test_report_lists_all_text_and_preserves_inputs(self):\n'
                     '        with tempfile.TemporaryDirectory() as directory:\n'
                     '            before, after = (pathlib.Path(directory) / name for name in '
                     "('before.json', 'after.json'))\n"
                     "            before.write_text(json.dumps({'*:A': 'old', '*:B': 'removed', '*:S': "
                     "'same'}))\n"
                     "            after.write_text(json.dumps({'*:A': 'new', '*:C': 'added', '*:S': "
                     "'same'}))\n"
                     '            original_bytes, current_bytes = (before.read_bytes(), after.read_bytes())\n'
                     '            report = taxonomy.knowledge_report(after, before)\n'
                     "            self.assertEqual(report['added'], {'*:C': 'added'})\n"
                     "            self.assertEqual(report['removed'], {'*:B': 'removed'})\n"
                     "            self.assertEqual(report['changed'], {'*:A': {'before': 'old', 'after': "
                     "'new'}})\n"
                     '            self.assertEqual(before.read_bytes(), original_bytes)\n'
                     '            self.assertEqual(after.read_bytes(), current_bytes)\n'
                     '            self.assertEqual(taxonomy.knowledge_report(after, before.parent / '
                     "'missing.json')['status'], 'original_missing')\n"
                     '\n'
                     '    def test_prompt_option_controls_detection_text(self):\n'
                     "        problem = checker.load_problem(checker._find_spec('lfsr'))\n"
                     "        history = [dict(code='module lfsr; endmodule', score=0.0, feedback='failed', "
                     "diagnostics={'signatures': ['X_BEFORE_FIRST_EDGE']})]\n"
                     '        default = checker.prompt_problem(problem, principles=True)\n'
                     "        plain = checker.build_prompt(default, history)[0]['content']\n"
                     '        verbose = checker.prompt_problem(problem, principles=True, '
                     'show_detect_text=True)\n'
                     "        full = checker.build_prompt(verbose, history)[0]['content']\n"
                     "        self.assertIn('Principle:', plain)\n"
                     "        self.assertNotIn('Detected signature', plain)\n"
                     "        self.assertNotIn('X_BEFORE_FIRST_EDGE', plain)\n"
                     "        self.assertIn('Detected signature X_BEFORE_FIRST_EDGE:', full)\n"
                     "        self.assertEqual(checker.prompt_audit(default)['selected_rules'], "
                     "checker.prompt_audit(verbose)['selected_rules'])\n"
                     '\n'
                     '    def test_cli_detection_option_and_missing_original_report(self):\n'
                     '        with contextlib.redirect_stdout(io.StringIO()) as output:\n'
                     '            with self.assertRaises(SystemExit) as result:\n'
                     "                checker.main(['--problem', 'lfsr', '--dut', str(checker.ROOT / "
                     "'candidates/bad/lfsr_syncreset.v'), '--principles', '--show-detect-text', "
                     "'--no-console-log'])\n"
                     '        self.assertEqual(result.exception.code, 1)\n'
                     "        self.assertIn('Detected signature', output.getvalue())\n"
                     '        with contextlib.redirect_stdout(io.StringIO()) as output:\n'
                     "            rc = checker.main(['--knowledge-report', '--knowledge-original', "
                     "'missing_original.json', '--no-console-log'])\n"
                     '        self.assertEqual(rc, 1)\n'
                     "        self.assertEqual(json.loads(output.getvalue())['status'], 'original_missing')\n"
                     "if __name__ == '__main__':\n"
                     '    unittest.main()'}


def _embedded_suite(pattern='test_*.py'):
    suite = unittest.TestSuite()
    for filename, source in EMBEDDED_TESTS.items():
        if not fnmatch.fnmatch(filename, pattern):
            continue
        module = types.ModuleType('_single_checker_' + filename.removesuffix('.py'))
        module.__file__ = __file__
        exec(compile(source, str(ROOT / ('checker.py[' + filename + ']')), 'exec'), module.__dict__)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    return suite


def _dispatch(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if '--init' in args:
        args.remove('--init')
        initialize_files()
        if not args:
            print('Resources ready in ' + str(ROOT))
            return 0
    if '--agent' in args:
        args.remove('--agent')
        if '--run-all' in args:
            if '--mode' not in args:
                args += ['--mode', 'agent']
        else:
            run_logged(lambda: _agent_main(args), args)
            return 0
    if '--help' in args or '-h' in args:
        print('Single-file modes: --agent (generate/check), --init (create resources).')
    return run_logged(lambda: _main(args), args)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    # Keep bundled paths and output inside this folder, including when launched
    # from elsewhere. Explicit relative input/output arguments retain their
    # meaning relative to the calling directory.
    caller = pathlib.Path.cwd()
    if caller != ROOT:
        paths = {'--spec', '--tb', '--ref', '--dut', '--taxonomy', '--knowledge',
                 '--knowledge-original', '--manifest', '--dut-dir', '--output-dir',
                 '--log-dir', '--diagnostics-json'}
        for index, arg in enumerate(args):
            option, equals, value = arg.partition('=')
            if option not in paths:
                continue
            if not equals and index + 1 < len(args):
                value = args[index + 1]
            if option == '--spec' and value in ('mine', 'bench'):
                continue
            path = pathlib.Path(value)
            if value and not path.is_absolute():
                value = str(caller / path)
                if equals:
                    args[index] = option + '=' + value
                else:
                    args[index + 1] = value
    with contextlib.chdir(ROOT):
        return _dispatch(args)


# Compatibility names for embedded regression tests, rather than separate scripts.
sys.modules.setdefault('checker', sys.modules[__name__])
sys.modules['agent'] = sys.modules[__name__]


if __name__ == '__main__':
    raise SystemExit(main())

