# Round-8 successor packaging notes

Round 8 repairs the exact independent rejection of `b54383b799f47fd1a849126d3f21fe6c643eb209`. The reviewer verdict is preserved verbatim and prior evidence is unchanged.

The verifier now rejects booleans before equality, rejects empty/vacuous trace structures, pins 789,115 numeric leaves and the immutable numeric-path digest, and applies duplicate/non-finite JSON rejection before any PASS-capable verifier path. Signed-envelope canonical-byte checks remain unchanged.

No protected execution, attack, signing, private-key search, or push was performed.
