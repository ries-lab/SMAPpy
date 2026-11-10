import warnings

import matplotlib.pyplot as plt

from decode_smlm.evaluation import evaluation, predict_dist
from decode_smlm.evaluation.evaluation import WeightedErrors


def log_kpi(loss_scalar: float, loss_cmp: dict, eval_set: dict, logger, step):

    logger.add_scalar('learning/test_ep', loss_scalar, step)

    assert loss_cmp.dim() >= 2
    for i in range(loss_cmp.size(1)):  # channel-wise mean
        logger.add_scalar('loss_cmp/test_ep_loss_ch_' + str(i), loss_cmp[:, i].mean(), step)

    logger.add_scalar_dict('eval/', eval_set, step)


def log_dists(tp, tp_match, pred, px_border, px_size, logger, step):

    """Log z vs z_gt"""
    f_x, ax_x = plt.subplots()
    f_y, ax_y = plt.subplots()
    f_z, ax_z = plt.subplots()
    f_phot, ax_phot = plt.subplots()

    predict_dist.emitter_deviations(tp, tp_match,
                                    px_border=px_border, px_size=px_size, axes=[ax_x, ax_y, ax_z, ax_phot])

    logger.add_figure('dist/x_offset', f_x, step)
    logger.add_figure('dist/y_offset', f_y, step)
    logger.add_figure('residuals/z_gt_pred', f_z, step)
    logger.add_figure('residuals/phot_gt_pred', f_phot, step)

    """Log prob dist"""
    f_prob, ax_prob = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predict_dist.utils.density_hist(pred.prob, ax_prob, bins=50)
    logger.add_figure('dist/prob', f_prob, step)


def log_train(*, loss_p_batch: (list, tuple), loss_mean: float, logger, step: int):

    logger.add_scalar('learning/train_ep', loss_mean, step)

    for i, loss_batch in enumerate(loss_p_batch):
        step_batch = step * len(loss_p_batch) + i
        if i % 10 != 0:
            continue

        logger.add_scalar('learning/train_batch', loss_batch, step_batch)


def post_process_log_test(*, loss_cmp, loss_scalar, x, y_out, y_tar, weight, em_tar,
                          px_border, px_size, post_processor, matcher, logger, step):

    """Post-Process"""
    em_out = post_processor.forward(y_out)

    """Match and Evaluate"""
    tp, fp, fn, tp_match = matcher.forward(em_out, em_tar)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evaluation.SMLMEvaluation(weighted_eval=WeightedErrors(mode='crlb', reduction='gaussian')).forward(tp, fp, fn, tp_match)

    """Log"""
    # DECODE logged figures of raw frames here (log_frames, with its plot package, which is not
    # ported): they went to TensorBoard only.  What a training shows is milestone (f)'s.

    # KPIs
    log_kpi(loss_scalar=loss_scalar, loss_cmp=loss_cmp, eval_set=result._asdict(), logger=logger, step=step)

    # distributions
    log_dists(tp=tp, tp_match=tp_match, pred=em_out, px_border=px_border, px_size=px_size, logger=logger, step=step)

    return
