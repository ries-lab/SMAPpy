"""The solvers: block Levenberg–Marquardt on a `Problem` (plan §5.6), and L-BFGS on its Gauss–Newton matrix
(plan §9 M7).  They know nothing about optics."""
from .blocks import BlockLM
from .cg import GaussNewtonCG
from .lbfgs import GaussNewtonLBFGS
from .lm import LevenbergMarquardt, schur_solve
from .problem import Normal, Problem
from .result import FitResult

__all__ = ["Problem", "Normal", "LevenbergMarquardt", "GaussNewtonLBFGS", "GaussNewtonCG", "BlockLM", "FitResult",
           "schur_solve"]
