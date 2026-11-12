"""netfit (a working name): SMAPpy's own network fitter, the successor of the DECODE port.

A network trained on simulated frames that detects and localizes molecules in one pass, written new
in SMAPpy's conventions -- x along the columns, pixel centres at whole numbers, calibrations as
SMAPpy holds them, SMAPpy's columns -- under SMAPpy's licence.  It takes no code from the GPL port
(`decode_smlm`); what it shares with DECODE are the published methods.  The plan is
`docs/decode.md` §10, the benchmark it is judged on `docs/decode/benchmark.md`.

Its network, simulation, loss, extraction and training are each a *part* named in its settings
(`settings.py`), so that one can be swapped and the result compared on the benchmark.  Version 1 is
DECODE's method as published, so that it can be checked against the port.

Importing this package loads no torch: the modules that need it import it themselves, as uipsf's
users do.
"""
