from functools import partial
from typing import Any, Callable, Literal

from decode.emitter import emitter
from decode.emitter import process

try:
    import skopt
except ImportError:
    skopt = None


def wrap_skopt_functional(
    opt: Literal["gp_minimize", "gbrt_minimize", "forest_minimize", "dummy_minimize"],
    **kwargs
):
    return partial(getattr(skopt, opt), **kwargs)


class FilterGenerator:
    def __init__(self, filter_attr_param: dict[str, Literal["less", "greater"]]):
        self._filter_attrs = filter_attr_param

    def factory(self, **kwargs) -> process.EmitterFilterGeneric:
        range = {
            k: process.range_factory(self._filter_attrs[k])(v)
            for k, v in kwargs.items()
        }
        return process.EmitterFilterGeneric(**range)


class Optimizer:
    def __init__(
        self,
        filter_fac: FilterGenerator,
        metric: Callable[[emitter.EmitterSet, emitter.EmitterSet], float],
        space: dict[str, skopt.space.Space],
        opt: Callable[[Callable, list[skopt.space.Space]], Any],
    ):
        """

        Args:
            filter_fac: filter factory
            metric: metric to optimize
            space: parameter space for optimization
            opt: optimizer backbone, typically from skopt, or wrapped version
        """
        self._em = None
        self._em_ref = None
        self._filter_fac = filter_fac
        self._metric = metric
        self._space = space
        self._space_anon = self._get_space_anon()
        self._opt = opt

    def _get_space_anon(self) -> list[skopt.space.Space]:
        space = [v for v in self._space.values()]
        # patch names
        for s, k in zip(space, self._space.keys()):
            s.name = k
        return space

    def objective(self, **kwargs) -> float:
        """
        Objective function for optimization.

        Args:
            **kwargs: parameter / value pair for filtering

        Example:
            >>> import skopt
            >>> opt = Optimizer(em, em_ref, {"xyz_sig_tot_nm": "less"}, metric)
            >>> space = [skopt.space.Real(20, 200, name="xyz_sig_tot_nm")]
            >>> obj = skopt.utils.use_named_args(space)(opt.objective)
            >>> res_gp = skopt.gp_minimize(obj, space, n_calls=20, random_state=42)
            # access the result
            >>> res_gp.x[0]

        Returns:

        """
        # construct filter
        f = self._filter_fac.factory(**kwargs)
        em = f.forward(self._em)
        return self._metric(em, self._em_ref)

    def fit(
        self, em: emitter.EmitterSet, em_ref: emitter.EmitterSet
    ) -> tuple[dict[str, Any], float]:
        """
        Get the best parameters for the metric.

        Args:
            em:
            em_ref:

        Returns:
            tuple[dict[str, Any], float]: best parameters, metric value
        """
        self._em = em
        self._em_ref = em_ref

        obj = skopt.utils.use_named_args(self._space_anon)(self.objective)
        res_gp = self._opt(obj, self._space_anon)
        return {
            k: v for k, v in zip(self._space.keys(), res_gp.x, strict=True)
        }, res_gp.fun
