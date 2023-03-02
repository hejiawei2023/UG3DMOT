# Author: hejw
import copy
import numpy as np
from utils.angle_difference import diff_orientation_correction

def js_distance(det, trk, dimensions):

    det_array = det["state"]
    det_cov = det["cov"]

    trk_array = trk.x_predict.flatten()[:7]
    trk_cov = trk.P_predict[:7, :7]

    m_states = (det_array + trk_array) / 2.0
    m_cov = (det_cov + trk_cov) / 4.0
    m = {"state": m_states, "cov": m_cov}
    p = {"state": det_array, "cov": det_cov}
    q = {"state": trk_array, "cov": trk_cov}

    diff = p["state"][3] - q["state"][3]
    corrected_yaw_diff = diff_orientation_correction(diff)
    cos_cost = 2 - np.cos(corrected_yaw_diff)
    result = (compute_kl(p, copy.copy(m), dimensions) + compute_kl(q, copy.copy(m), dimensions)) * 0.5 * cos_cost

    return result


def kl_distance(det, trk, dimensions):
    det_array = det["state"]
    det_cov = det["cov"]

    trk_array = trk.x_predict.flatten()[:7]
    trk_cov = trk.P_predict[:7, :7]

    p = {"state": det_array, "cov": det_cov}
    q = {"state": trk_array, "cov": trk_cov}

    diff = p["state"][3] - q["state"][3]
    corrected_yaw_diff = diff_orientation_correction(diff)
    cos_cost = 2 - np.cos(corrected_yaw_diff)
    result = compute_kl(q, p, dimensions) * 0.5 * cos_cost
    return result

def compute_kl(p, q, dimensions=None):
    # kl(p||q) = 0.5 * [log(|cov_q| / |cov_p|) - len(dimensions) + trace + diff @ inverse_cov_q @ diff.T]
    # trace = tr{inverse_cov_q * cov_p}
    # diff = μp - μq
    if dimensions is None:
        k = len(p["state"])

    else:
        p["state"] = p["state"][dimensions]
        p["cov"] = p["cov"][dimensions, :][:, dimensions]
        q["state"] = q["state"][dimensions]
        q["cov"] = q["cov"][dimensions, :][:, dimensions]
        k = len(dimensions)

    diff = (p["state"] - q["state"])

    if 3 in dimensions:
        corrected_yaw_diff = diff_orientation_correction(diff[3])
        diff[3] = corrected_yaw_diff

    modules_p = np.linalg.det(p["cov"])
    modules_q = np.linalg.det(q["cov"])
    inverse_q = np.linalg.inv(q["cov"])
    trace = (inverse_q @ p["cov"]).trace()
    cost = np.log(modules_q / modules_p) - k + trace + diff @ inverse_q @ diff.T
    return cost * 0.5
