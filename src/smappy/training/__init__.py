"""Training a fitter on simulated frames: the parts that are SMAPpy's, not a method's.

A deep-learning fitter (DECODE now, others later: docs/decode.md, docs/decode/poc-plan.md §5)
learns from frames it simulates, so it is prepared on the data it will fit: a short fit of the
data says what to simulate (`smappy.training.prefit`), and a job folder holds everything a
training needs, so it can be handed to another machine and run there (`smappy.training.job`).
Nothing here imports torch or a method's package: the GUI builds the form, checks a folder and
follows a training's progress without either.
"""
