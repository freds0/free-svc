# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter

import numpy as np
from tabulate import tabulate


def comp_purity(p_xy, axis):
    max_p = p_xy.max(axis=axis)
    marg_p = p_xy.sum(axis=axis)
    indv_pur = max_p / marg_p
    aggr_pur = max_p.sum()
    return indv_pur, aggr_pur


def comp_entropy(p):
    return (-p * np.log(p + 1e-8)).sum()


def comp_norm_mutual_info(p_xy):
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    pmi = np.log(p_xy / np.matmul(p_x, p_y) + 1e-8)
    mi = (p_xy * pmi).sum()
    h_x = comp_entropy(p_x)
    h_y = comp_entropy(p_y)
    return mi, mi / h_x, mi / h_y, h_x, h_y


def pad(labs, n):
    if n == 0:
        return np.array(labs)
    return np.concatenate([[labs[0]] * n, labs, [labs[-1]] * n])


def comp_avg_seg_dur(labs_list):
    n_frms = 0
    n_segs = 0
    for labs in labs_list:
        labs = np.array(labs)
        edges = np.zeros(len(labs)).astype(bool)
        edges[0] = True
        edges[1:] = labs[1:] != labs[:-1]
        n_frms += len(edges)
        n_segs += edges.astype(int).sum()
    return n_frms / n_segs


def comp_joint_prob(uid2refs, uid2hyps):
    cnts = Counter()
    skipped = []
    abs_frmdiff = 0
    for uid in uid2refs:
        if uid not in uid2hyps:
            skipped.append(uid)
            continue
        refs = uid2refs[uid]
        hyps = uid2hyps[uid]
        abs_frmdiff += abs(len(refs) - len(hyps))
        min_len = min(len(refs), len(hyps))
        refs = refs[:min_len]
        hyps = hyps[:min_len]
        cnts.update(zip(refs, hyps))
    tot = sum(cnts.values())

    ref_set = sorted({ref for ref, _ in cnts.keys()})
    hyp_set = sorted({hyp for _, hyp in cnts.keys()})
    ref2pid = dict(zip(ref_set, range(len(ref_set))))
    hyp2lid = dict(zip(hyp_set, range(len(hyp_set))))

    p_xy = np.zeros((len(ref2pid), len(hyp2lid)), dtype=float)
    for (ref, hyp), cnt in cnts.items():
        p_xy[ref2pid[ref], hyp2lid[hyp]] = cnt
    freq_xy = p_xy
    full_freq_xy = np.zeros((len(ref2pid), 4096), dtype=float)
    for (ref, hyp), cnt in cnts.items():
        full_freq_xy[ref2pid[ref], int(hyp)] = cnt
    p_xy = p_xy / p_xy.sum()
    return (
        freq_xy,
        full_freq_xy,
        p_xy,
        ref2pid,
        hyp2lid,
        tot,
        abs_frmdiff,
        skipped,
        ref_set,
        hyp_set,
    )


def comp_phone2code(p_xy):
    p_x = p_xy.sum(axis=1, keepdims=True)  # ref (phone)
    p_y = p_xy.sum(axis=0, keepdims=True)  # hyp (code)

    p_x_y = p_xy / p_y  # P(x | y) = P(phone | code)

    y_order = np.argsort(p_x_y.argmax(0))
    p_x_y_sorted_y = np.take_along_axis(p_x_y, y_order.reshape((1, -1)), axis=1)

    x_order = np.argsort(p_x[:, 0])
    x_order = np.flip(x_order)
    p_x_y_sorted_x = np.take_along_axis(p_x_y, x_order.reshape((-1, 1)), axis=0)
    y_order = np.argsort(p_x_y_sorted_x.argmax(0))
    p_x_y_sorted_xy = np.take_along_axis(
        p_x_y_sorted_x, y_order.reshape((1, -1)), axis=1
    )

    return p_x_y, p_x_y_sorted_xy, p_x_y_sorted_y, x_order


def compute_show_pnmi(uid2refs, uid2hyps, upsample=1, show_results: bool = False):
    for k, v in uid2hyps.items():
        uid2hyps[k] = pad(v, 0).repeat(upsample)

    (
        freq_xy,
        full_freq_xy,
        p_xy,
        ref2pid,
        hyp2lid,
        tot,
        frmdiff,
        skipped,
        ref_set,
        hyp_set,
    ) = comp_joint_prob(uid2refs, uid2hyps)
    ref_pur_by_hyp, ref_pur = comp_purity(p_xy, axis=0)
    hyp_pur_by_ref, hyp_pur = comp_purity(p_xy, axis=1)
    (mi, mi_norm_by_ref, mi_norm_by_hyp, h_ref, h_hyp) = comp_norm_mutual_info(p_xy)

    if show_results:
        print(
            tabulate(
                [[hyp_pur, ref_pur, mi_norm_by_ref]],
                ["Cls Pur", "Phn Pur", "PNMI"],
                floatfmt=".3f",
                tablefmt="fancy_grid",
            )
        )

    return {
        "cls_pur": hyp_pur,
        "phn_pur": ref_pur,
        "pnmi": mi_norm_by_ref,
    }


def compute_snmi(p_xy):
    _, ref_pur = comp_purity(p_xy, axis=0)
    _, hyp_pur = comp_purity(p_xy, axis=1)
    (_, mi_norm_by_ref, _, _, _) = comp_norm_mutual_info(p_xy)

    return {
        "cls_pur": hyp_pur,
        "spk_pur": ref_pur,
        "snmi": mi_norm_by_ref,
    }
