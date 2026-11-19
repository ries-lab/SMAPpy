import warnings
from collections import namedtuple
from dataclasses import dataclass
from typing import Union, Optional, Callable, Literal

import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import torch
import torchmetrics

from .metric import (
    precision,
    recall,
    jaccard,
    efficiency,
    f1,
    bootstrap_counts_se,
    bootstrap_matching_se,
    Metric,
)
from .metric import rmse, mae, mad, rmse_generic
from ..emitter import EmitterSet
from ..emitter import process
from . import match_emittersets


@dataclass
class ClassifiedEmitters:
    tp: EmitterSet
    fp: EmitterSet
    fn: EmitterSet


@dataclass
class Classification:
    prec: Union[float, torch.Tensor]
    rec: Union[float, torch.Tensor]
    jac: Union[float, torch.Tensor]
    f1: Union[float, torch.Tensor]


class DetectionEvaluation:
    _seg_eval_return = namedtuple(
        "seg_eval", ["prec", "rec", "jac", "f1", "n_tar", "n_out", "n_tp"]
    )

    def __init__(self):
        self._tp = None
        self._fp = None
        self._fn = None
        self._prec = None
        self._rec = None
        self._jac = None
        self._f1 = None

    def __str__(self):
        if self._tp is None or self._fp is None or self._fn is None:
            return "Segmentation evaluation unavailable. Run .forward(tp, fp, fn)"

        actual_em = len(self._tp) + len(self._fn)
        pred_em = len(self._tp) + len(self._fp)

        str_repr = "Segmentation evaluation (cached values)\n"
        str_repr += (
            f"Number of actual emitters: {actual_em} Predicted emitters: {pred_em}\n"
        )
        str_repr += (
            f"Number of TP: {len(self._tp)} FP: {len(self._fp)} FN: {len(self._fn)}\n"
        )
        str_repr += f"Jaccard: {self._jac.to_str()}\n"
        str_repr += f"F1Score: {self._f1.to_str()}\n"
        str_repr += f"Precision: {self._prec.to_str()}, Recall: {self._rec.to_str()}\n"

        return str_repr

    def forward(
        self,
        tp: EmitterSet,
        fp: EmitterSet,
        fn: EmitterSet,
        se: bool = False,
        se_fn: Literal["bootstrap_se"] | Callable = "bootstrap_se",
    ):
        """
        Forward emitters through evaluation.

        Args:
            tp: true positives
            fp: false positives
            fn: false negatives
            se: whether to compute standard error (bootstrapped). Only for precision, recall, Jaccard, and F1.
            se_fn: function to compute standard errors.
             This function takes the metric function, tp, fp, fn as inputs, and outputs standard errors.
             By default, bootstrapping.

        Returns:
            prec (Metric): precision value [, standard error]
            rec (Metric): recall value [, standard error]
            jac (Metric): jaccard value [, standard error]
            f1 (Metric): f1 score value [, standard error]

        """
        if se_fn == "bootstrap_se":
            se_fn = bootstrap_counts_se
        prec = Metric(
            mean=precision(len(tp), len(fp)),
            se=se_fn(precision, tp=len(tp), fp=len(fp)) if se else None,
        )
        rec = Metric(
            mean=recall(len(tp), len(fn)),
            se=se_fn(recall, tp=len(tp), fn=len(fn)) if se else None,
        )
        jac = Metric(
            mean=jaccard(len(tp), len(fp), len(fn)),
            se=se_fn(jaccard, tp=len(tp), fp=len(fp), fn=len(fn)) if se else None,
        )
        f1_score = Metric(
            mean=f1(len(tp), len(fp), len(fn)),
            se=se_fn(f1, tp=len(tp), fp=len(fp), fn=len(fn)) if se else None,
        )

        # store last result to cache
        self._tp, self._fp, self._fn = tp, fp, fn
        self._prec, self._rec, self._jac, self._f1 = (prec, rec, jac, f1_score)

        return self._seg_eval_return(
            prec=prec,
            rec=rec,
            jac=jac,
            f1=f1_score,
            n_tar=len(tp) + len(fn),
            n_out=len(tp) + len(fp),
            n_tp=len(tp),
        )


class DistanceEvaluation:
    """
    A small wrapper calss that holds distance evaluations and accepts sets of emitters as inputs.
    """

    _dist_eval_return = namedtuple(
        "dist_eval",
        [
            "rmse_lat",
            "rmse_ax",
            "rmse_vol",
            "mae_lat",
            "mae_ax",
            "mae_vol",
            "mad_lat",
            "mad_ax",
            "mad_vol",
            "acc_code",
            "rmse_phot",
            "rmse_phot_pnorm"
        ],
    )

    def __init__(self, num_codes: Optional[int] = None):
        self._rmse_lat = None
        self._rmse_ax = None
        self._rmse_vol = None

        self._mae_lat = None
        self._mae_ax = None
        self._mae_vol = None

        self._mad_lat = None
        self._mad_ax = None
        self._mad_vol = None

        self._acc_code = None
        self._acc_code_impl = None
        if num_codes is not None:
            self._acc_code_impl = torchmetrics.Accuracy(
                num_classes=num_codes,
                task="multiclass" if num_codes >= 2 else "binary",
            )

        self._rmse_phot = None
        self._rmse_phot_pnorm = None

    def __str__(self):
        if self._rmse_lat is None:
            return "Distance Evaluation unavailable. Run .forward(tp, tp_match)."

        str_repr = "Distance Evaluation (cached values)\n"
        str_repr += f"RMSE: Lat. {self._rmse_lat.to_str()} Axial. {self._rmse_ax.to_str()} Vol. {self._rmse_vol.to_str()}\n"
        str_repr += f"MAE: Lat. {self._mae_lat.to_str()} Axial. {self._mae_ax.to_str()} Vol. {self._mae_vol.to_str()}\n"
        str_repr += f"MAD: Lat. {self._mad_lat.to_str()} Axial. {self._mad_ax.to_str()} Vol. {self._mad_vol.to_str()}\n"

        return str_repr

    def forward(
        self,
        tp: EmitterSet,
        tp_match: EmitterSet,
        se: bool = False,
        se_fn: Literal["bootstrap_se"] | Callable = "bootstrap_se",
    ):
        """

        Args:
            tp: true positives
            tp_match: matching ground truths
            se: whether to compute standard error (bootstrapped). Only for RMSE, MAE, and MAD.
            se_fn: function to compute standard errors.
             This function takes the metric function, predictions, and ground truth as inputs, and outputs standard errors.
             By default, bootstrapping.

        Returns:
            rmse_lat: RMSE lateral
            rmse_ax: RMSE axial
            rmse_vol: RMSE volumetric
            mae_lat: MAE lateral
            mae_ax: MAE axial
            mae_vol: MAE volumetric
            mad_lat: MAD lateral
            mad_ax: MAD axial
            mad_vol: MAD volumetric

        """

        rmse_lat, rmse_axial, rmse_vol = rmse(tp.xyz_nm, tp_match.xyz_nm)
        mae_lat, mae_axial, mae_vol = mae(tp.xyz_nm, tp_match.xyz_nm)
        mad_lat, mad_axial, mad_vol = mad(tp.xyz_nm, tp_match.xyz_nm)

        rmse_phot = rmse_generic(tp.phot, tp_match.phot).tolist()
        rmse_phot_pnorm = rmse_generic(tp.phot, tp_match.phot, re_norm=tp_match.phot.sqrt()).tolist()

        if se:
            if se_fn == "bootstrap_se":
                se_fn = bootstrap_matching_se
            rmse_lat_se, rmse_axial_se, rmse_vol_se = se_fn(
                rmse, xyz=tp.xyz_nm, xyz_ref=tp_match.xyz_nm
            )
            mae_lat_se, mae_axial_se, mae_vol_se = se_fn(
                mae, xyz=tp.xyz_nm, xyz_ref=tp_match.xyz_nm
            )
            mad_lat_se, mad_axial_se, mad_vol_se = se_fn(
                mad, xyz=tp.xyz_nm, xyz_ref=tp_match.xyz_nm
            )
        else:
            rmse_lat_se, rmse_axial_se, rmse_vol_se = None, None, None
            mae_lat_se, mae_axial_se, mae_vol_se = None, None, None
            mad_lat_se, mad_axial_se, mad_vol_se = None, None, None

        rmse_lat = Metric(rmse_lat, rmse_lat_se)
        rmse_axial = Metric(rmse_axial, rmse_axial_se)
        rmse_vol = Metric(rmse_vol, rmse_vol_se)
        mae_lat = Metric(mae_lat, mae_lat_se)
        mae_axial = Metric(mae_axial, mae_axial_se)
        mae_vol = Metric(mae_vol, mae_vol_se)
        mad_lat = Metric(mad_lat, mad_lat_se)
        mad_axial = Metric(mad_axial, mad_axial_se)
        mad_vol = Metric(mad_vol, mad_vol_se)

        # store in cache
        self._rmse_lat, self._rmse_ax, self._rmse_vol = rmse_lat, rmse_axial, rmse_vol
        self._mae_lat, self._mae_ax, self._mae_vol = mae_lat, mae_axial, mae_vol
        self._mad_lat, self._mad_ax, self._mad_vol = mad_lat, mad_axial, mad_vol
        self._rmse_phot = rmse_phot
        self._rmse_phot_pnorm = rmse_phot_pnorm

        # accuracy code
        acc_code = float("nan")
        if self._acc_code_impl is not None:
            if len(tp) != 0:
                acc_code = Metric(self._acc_code_impl(tp.code, tp_match.code).item())
            self._acc_code_impl.reset()

        return self._dist_eval_return(
            rmse_lat=rmse_lat,
            rmse_ax=rmse_axial,
            rmse_vol=rmse_vol,
            mae_lat=mae_lat,
            mae_ax=mae_axial,
            mae_vol=mae_vol,
            mad_lat=mad_lat,
            mad_ax=mad_axial,
            mad_vol=mad_vol,
            acc_code=acc_code,
            rmse_phot=rmse_phot,
            rmse_phot_pnorm=rmse_phot_pnorm,
        )


class WeightedErrors:
    _modes_all = ("phot", "crlb")
    _reduction_all = ("mstd", "gaussian")
    _return = namedtuple(
        "weighted_err",
        ["dxyz_red", "dphot_red", "dbg_red", "dxyz_w", "dphot_w", "dbg_w"],
    )

    def __init__(self, mode, reduction):
        self.mode = mode
        self.reduction = reduction

        if self.mode not in self._modes_all:
            raise ValueError(
                f"Mode {self.mode} not implemented. Available modes are {self._modes_all}"
            )

        if self.reduction not in self._reduction_all:
            raise ValueError(
                f"Reduction type {self.reduction} not implemented. Available reduction types"
                f"are {self._reduction_all}."
            )

    @staticmethod
    def _reduce(dxyz: torch.Tensor, dphot: torch.Tensor, dbg: torch.Tensor, reduction):
        """
        Reduce the weighted errors as by the specified method.

        Args:
            dxyz (torch.Tensor): weighted err in xyz, N x 3
            dphot (torch.Tensor): weighted err in phot, N
            dbg (torch.Tensor): weighted err in bg, N
            reduction (string,None): reduction type

        Returns:
            (torch.Tensor or tuple of tensors)

        """

        def norm_fit_nan(input_data, warning=True):
            try:
                out = scipy.stats.norm.fit(input_data)
                out = torch.tensor(out)

            except RuntimeError:
                warnings.warn("Non-Finite values encountered during fitting.")
                out = float("nan") * torch.ones(2)

            return out

        if reduction == "mstd":
            return (
                (dxyz.mean(0), dxyz.std(0)),
                (dphot.mean(), dphot.std()),
                (dbg.mean(), dbg.std()) if dbg is not None else None,
            )

        elif reduction == "gaussian":
            dxyz_mu_sig = torch.stack([norm_fit_nan(dxyz[:, i]) for i in range(3)], 0)
            dphot_mu_sig = norm_fit_nan(dphot)
            dbg_mu_sig = (norm_fit_nan(dbg) if dbg is not None else None,)

            return (
                (dxyz_mu_sig[:, 0], dxyz_mu_sig[:, 1]),
                (dphot_mu_sig[0], dphot_mu_sig[1]),
                (dbg_mu_sig[0], dbg_mu_sig[1]) if dbg is not None else None,
            )

        else:
            raise ValueError

    @staticmethod
    def plot_error(dxyz, dphot, dbg, axes=None):
        """
        Plot the histograms

        Args:
            dxyz (torch.Tensor): weighted err in xyz, N x 3
            dphot (torch.Tensor): weighted err in phot, N
            dbg (torch.Tensor): weighted err in bg, N
            axes (tuple of axes,None): axes to which to plot to, tuple of size 6 or None

        Returns:
            axes

        """

        if axes is None:
            _, axes = plt.subplots(5)
            # axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]

        else:
            if len(axes) != 5:
                raise ValueError("You must parse exactly 6 axes objects or None.")

        if len(dxyz) == 0:
            return axes

        if len(dxyz[:, 0]) != len(dphot) or len(dphot) != len(dbg):
            raise ValueError("Inconsistent number of elements.")

        sns.distplot(
            dxyz[:, 0].numpy(),
            norm_hist=True,
            kde=False,
            fit=scipy.stats.norm,
            ax=axes[0],
        )
        sns.distplot(
            dxyz[:, 1].numpy(),
            norm_hist=True,
            kde=False,
            fit=scipy.stats.norm,
            ax=axes[1],
        )
        sns.distplot(
            dxyz[:, 2].numpy(),
            norm_hist=True,
            kde=False,
            fit=scipy.stats.norm,
            ax=axes[2],
        )

        sns.distplot(
            dphot.numpy(), norm_hist=True, kde=False, fit=scipy.stats.norm, ax=axes[3]
        )
        sns.distplot(
            dbg.numpy(), norm_hist=True, kde=False, fit=scipy.stats.norm, ax=axes[4]
        )

        return axes

    def forward(
        self, tp: EmitterSet, ref: EmitterSet, plot: bool = False, axes=None
    ) -> namedtuple:
        """

        Args:
            tp (EmitterSet): true positives
            ref (EmitterSet): matching ground truth
            plot (bool): plot histograms
            axes (list,tuple): axis to which to plot the histograms

        Returns:

        """

        if len(tp) != len(ref):
            raise ValueError(
                f"Size of true positives ({len(tp)}) does not match size of reference ({len(ref)})."
            )

        dxyz = tp.xyz_nm - ref.xyz_nm
        dphot = tp.phot - ref.phot
        dbg = tp.bg - ref.bg if tp.bg is not None else None

        if self.mode == "phot":
            # definition of the 0st / 1st order approximations for the sqrt cramer rao
            xyz_scr_est = 1 / ref.phot.unsqueeze(1).sqrt()
            phot_scr_est = ref.phot.sqrt()
            bg_scr_est = ref.bg.sqrt() if ref.bg is not None else None

            dxyz_w = dxyz / xyz_scr_est
            dphot_w = dphot / phot_scr_est
            dbg_w = dbg / bg_scr_est if dbg is not None else None

        elif self.mode == "crlb":
            dxyz_w = dxyz / ref.xyz_scr_nm if ref.xyz_scr is not None else None
            dphot_w = dphot / ref.phot_scr if ref.phot_scr is not None else None
            dbg_w = dbg / ref.bg_scr if dbg is not None else None

        else:
            raise ValueError

        if plot:
            _ = self.plot_error(dxyz_w, dphot_w, dbg_w, axes=axes)

        dxyz_wred, dphot_wred, dbg_wred = self._reduce(
            dxyz_w, dphot_w, dbg_w, reduction=self.reduction
        )
        return self._return(
            dxyz_red=dxyz_wred,
            dphot_red=dphot_wred,
            dbg_red=dbg_wred,
            dxyz_w=dxyz_w,
            dphot_w=dphot_w,
            dbg_w=dbg_w,
        )


class SMLMEvaluation:
    alpha_lat = 1  # nm
    alpha_ax = 0.5  # nm

    _return = namedtuple(
        "eval_set",
        [
            "prec",
            "rec",
            "jac",
            "f1",
            "effcy_lat",
            "effcy_ax",
            "effcy_vol",
            "rmse_lat",
            "rmse_ax",
            "rmse_vol",
            "mae_lat",
            "mae_ax",
            "mae_vol",
            "mad_lat",
            "mad_ax",
            "mad_vol",
            "acc_code",
            "rmse_phot",
            "rmse_phot_pnorm",
            "dx_red_mu",
            "dx_red_sig",
            "dy_red_mu",
            "dy_red_sig",
            "dz_red_mu",
            "dz_red_sig",
            "dphot_red_mu",
            "dphot_red_sig",
        ],
    )

    descriptors = {
        "prec": "Precision",
        "rec": "Recall",
        "jac": "Jaccard Index",
        "rmse_lat": "RMSE lateral",
        "rmse_ax": "RMSE axial",
        "rmse_vol": "RMSE volumetric",
        "mae_lat": "MAE lateral",
        "mae_ax": "MAE axial",
        "mae_vol": "MAE volumetric",
        "mad_lat": "Mean average distance lateral",
        "mad_ax": "Mean average distance axial",
        "mad_vol": "Mean average distance in 3D",
        "acc_code": "Accuracy code",
        "rmse_phot": "RMSE photon count",
        "rmse_phot_pnorm": "RMSE photon count (normalised by sqrt(phot of tar)",
        "dx_red_sig": "CRLB normalised error in x",
        "dy_red_sig": "CRLB normalised error in y",
        "dz_red_sig": "CRLB normalised error in z",
        "dx_red_mu": "CRLB normalised bias in x",
        "dy_red_mu": "CRLB normalised bias in y",
        "dz_red_mu": "CRLB normalised bias in z",
        "dphot_red_mu": "CRLB normalised bias in photon count",
        "dphot_red_sig": "CRLB normalised error in photon count",
        "f1": "F1-Score",
        # property attributes
        "effcy_lat": "Efficiency lateral",
        "effcy_ax": "Efficiency axial",
        "effcy_vol": "Efficiency volumetric",
    }

    def __init__(
        self,
        seg_eval=DetectionEvaluation(),
        dist_eval=DistanceEvaluation(),
        weighted_eval=WeightedErrors(mode="crlb", reduction="gaussian"),
    ):
        """
        A wrapper class to combine things into one.
        """
        self.seg_eval = seg_eval
        self.dist_eval = dist_eval
        self.weighted_eval = weighted_eval

        self.prec = None
        self.rec = None
        self.jac = None
        self.f1 = None

        self.rmse_vol = None
        self.rmse_lat = None
        self.rmse_ax = None
        self.mae_vol = None
        self.mae_lat = None
        self.mae_ax = None
        self.mad_vol = None
        self.mad_lat = None
        self.mad_ax = None

        self.acc_code = None
        self.rmse_phot = None
        self.rmse_phot_pnorm = None

    @property
    def effcy_lat(self):
        return Metric(efficiency(self.jac.mean, self.rmse_lat.mean, self.alpha_lat))

    @property
    def effcy_ax(self):
        return Metric(efficiency(self.jac.mean, self.rmse_ax.mean, self.alpha_ax))

    @property
    def effcy_vol(self):
        return Metric((self.effcy_lat + self.effcy_ax) / 2)

    def __str__(self):
        str = "------------------------ Evaluation Set ------------------------\n"
        str += "Precision {}\n".format(self.prec.to_str())
        str += "Recall {}\n".format(self.rec.to_str())
        str += "Jaccard {}\n".format(self.jac.to_str())
        str += "F1Score {}\n".format(self.f1.to_str())
        str += "RMSE lat. {}\n".format(self.rmse_lat.to_str())
        str += "RMSE ax. {}\n".format(self.rmse_ax.to_str())
        str += "RMSE vol. {}\n".format(self.rmse_vol.to_str())
        str += "MAE lat. {}\n".format(self.mae_lat.to_str())
        str += "MAE ax. {}\n".format(self.mae_ax.to_str())
        str += "MAE vol. {}\n".format(self.mae_vol.to_str())
        str += "MAD lat. {}\n".format(self.mad_lat.to_str())
        str += "MAD ax. {}\n".format(self.mad_ax.to_str())
        str += "MAD vol. {}\n".format(self.mad_vol.to_str())
        str += "Efficiency lat. {}\n".format(self.effcy_lat.to_str())
        str += "Efficiency ax. {}\n".format(self.effcy_ax.to_str())
        str += "Efficiency vol. {}\n".format(self.effcy_vol.to_str())
        str += "-----------------------------------------------------------------"
        return str

    def forward(
        self,
        tp,
        fp,
        fn,
        p_ref,
        se: bool = False,
        se_fn_matching: Literal["bootstrap_se"] | Callable = "bootstrap_se",
        se_fn_counts: Literal["bootstrap_se"] | Callable = "bootstrap_se",
    ) -> _return:
        """
        Evaluate sets of emitters by all available metrics.

        Args:
            tp: true positives
            fp: false positives
            fn: false negatives
            p_ref: true positive references (i.e. the ground truth that has been matched to tp)
            se: whether to return the standard errors (only for precision, recall, Jaccard, F1, RMSE, MAE, MAD)
            se_fn_matching: function to compute standard errors.
             This function takes the metric function, predictions, and ground truth as inputs, and outputs standard errors.
             By default, bootstrapping.
            se_fn_matching: function to compute standard errors.
             This function takes the metric function, tp, fp, fn as inputs, and outputs standard errors.
             By default, bootstrapping.

        Returns:
            namedtuple: A namedtuple of floats containing

                - **prec** (*decode.evaluation.metric.Metric*): Precision
                - **rec** (*decode.evaluation.metric.Metric*): Recall
                - **jac** (*decode.evaluation.metric.Metric*): Jaccard
                - **f1** (*decode.evaluation.metric.Metric*): F1-Score
                - **effcy_lat** (*decode.evaluation.metric.Metric*): Efficiency lateral
                - **effcy_ax** (*decode.evaluation.metric.Metric*): Efficiency axial
                - **effcy_vol** (*decode.evaluation.metric.Metric*): Efficiency volumetric
                - **rmse_lat** (*decode.evaluation.metric.Metric*): RMSE lateral
                - **rmse_ax** (*decode.evaluation.metric.Metric*): RMSE axial
                - **rmse_vol** (*decode.evaluation.metric.Metric*): RMSE volumetric
                - **mae_lat** (*decode.evaluation.metric.Metric*): MAE lateral
                - **mae_ax** (*decode.evaluation.metric.Metric*): MAE axial
                - **mae_vol** (*decode.evaluation.metric.Metric*): MAE volumetric
                - **mad_lat** (*decode.evaluation.metric.Metric*): MAD lateral
                - **mad_ax** (*decode.evaluation.metric.Metric*): MAD axial
                - **mad_vol** (*decode.evaluation.metric.Metric*): MAD volumetric


        """
        seg_out = self.seg_eval.forward(tp, fp, fn, se=se, se_fn=se_fn_counts)
        [
            setattr(self, k, v)
            for k, v in seg_out._asdict().items()
            if k in ["prec", "rec", "jac", "f1"]
        ]

        dist_out = self.dist_eval.forward(tp, p_ref, se=se, se_fn=se_fn_matching)
        [setattr(self, k, v) for k, v in dist_out._asdict().items()]

        # weight_out = self.weighted_eval.forward(tp, p_ref, plot=False)
        # dx_red = (weight_out.dxyz_red[0][0].item(), weight_out.dxyz_red[1][0].item())
        # dy_red = (weight_out.dxyz_red[0][1].item(), weight_out.dxyz_red[1][1].item())
        # dz_red = (weight_out.dxyz_red[0][2].item(), weight_out.dxyz_red[1][2].item())None,  #

        return self._return(
            prec=seg_out.prec,
            rec=seg_out.rec,
            jac=seg_out.jac,
            f1=seg_out.f1,
            effcy_lat=self.effcy_lat,
            effcy_ax=self.effcy_ax,
            effcy_vol=self.effcy_vol,
            rmse_lat=dist_out.rmse_lat,
            rmse_ax=dist_out.rmse_ax,
            rmse_vol=dist_out.rmse_vol,
            mae_lat=dist_out.mae_lat,
            mae_ax=dist_out.mae_ax,
            mae_vol=dist_out.mae_vol,
            mad_lat=dist_out.mad_lat,
            mad_ax=dist_out.mad_ax,
            mad_vol=dist_out.mad_vol,
            acc_code=dist_out.acc_code,
            rmse_phot=dist_out.rmse_phot,
            rmse_phot_pnorm=dist_out.rmse_phot_pnorm,
            dx_red_mu=Metric(float("nan")),  # dx_red[0],
            dx_red_sig=Metric(float("nan")),  # dx_red[1],
            dy_red_mu=Metric(float("nan")),  # dy_red[0],
            dy_red_sig=Metric(float("nan")),  # dy_red[1],
            dz_red_mu=Metric(float("nan")),  # dz_red[0],
            dz_red_sig=Metric(float("nan")),  # dz_red[1],
            dphot_red_mu=Metric(float("nan")),  # weight_out.dphot_red[0].item(),
            dphot_red_sig=Metric(float("nan")),  # weight_out.dphot_red[1].item(),
        )._asdict()


class EvaluationSMLM:  #  ToDo: Better name
    def __init__(
        self,
        matcher: match_emittersets.EmitterMatcher,
        eval_matched: DistanceEvaluation,
        em_filter: Optional[process.EmitterProcess] = None,
    ):
        self._matcher = matcher
        self._em_filter = em_filter
        self._eval_matched = eval_matched
        self._eval_loose = DetectionEvaluation()

    def forward(self, em: EmitterSet, em_ref: EmitterSet) -> dict:
        em = self._em_filter.forward(em) if self._em_filter is not None else em
        tp, fp, fn, tp_match = self._matcher.forward(em, em_ref)

        metrics = dict()
        metrics.update(self._eval_loose.forward(tp=tp, fp=fp, fn=fn)._asdict())
        metrics.update(self._eval_matched.forward(tp=tp, tp_match=tp_match)._asdict())

        return metrics
