"""Unit tests. They run the notebook's own function and class definitions (no data folder needed)."""
import itertools
import json
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cells():
    nb = json.load(open(os.path.join(ROOT, "hmm_activity_recognition_final.ipynb"), encoding="utf8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.fixture(scope="module")
def ns():
    cells, g = _cells(), {}
    pick = lambda marker: next(c for c in cells if marker in c)
    cwd = os.getcwd(); os.chdir(ROOT)
    try:
        exec(pick("import os, re, glob"), g)                                          # imports and configuration
        exec(pick("FNAME = re.compile").split("train_recs = load_dir")[0], g)         # loading functions
        exec(pick("FEATURES = [").split('for r in train_recs + test_recs: r["F"]')[0], g)   # feature functions
        exec(pick("class GaussianHMM").split("\nXs, ys")[0], g)                       # HMM class
        exec(pick("def evaluate").split("\nres = {}")[0], g)                          # evaluation
    finally:
        os.chdir(cwd)
    return g


def test_to_seconds_units(ns):
    t = np.arange(5) * 0.01
    for scale in (1, 1e3, 1e9):
        assert np.allclose(ns["to_seconds"](t * scale + 5 * scale), t)


def test_load_recording_resamples_and_trims(ns, tmp_path):
    t = np.arange(0, 8, 0.01)                              # 100 Hz, 8 s
    df = pd.DataFrame({"timestamp": t, "ax": 0.0, "ay": 0.0, "az": -9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0})
    df.to_csv(tmp_path / "still_01.csv", index=False)
    out = ns["load_recording"](tmp_path / "still_01.csv")
    x, fs_raw = out[0], out[1]
    assert fs_raw == pytest.approx(100, rel=0.02)
    assert x.shape[1] == 6 and np.isfinite(x).all()
    assert len(x) == pytest.approx((8 - ns["TRIM_START_S"] - ns["TRIM_END_S"]) * ns["FS"], abs=3)
    assert np.allclose(x[:, 2], -9.81)                      # no zero-padding bias at the edges


def test_load_recording_missing_column(ns, tmp_path):
    pd.DataFrame({"timestamp": [0, 1], "ax": [0, 0]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError):
        ns["load_recording"](tmp_path / "bad.csv")


def test_window_features_dominant_frequency(ns):
    t = np.arange(ns["WIN"]) / ns["FS"]
    w = np.zeros((ns["WIN"], 6)); w[:, 2] = 9.81 + np.sin(2 * np.pi * 2.0 * t)
    f = dict(zip(ns["FEATURES"], ns["window_features"](w)))
    assert f["acc_dom_freq"] == pytest.approx(2.0)
    assert len(ns["window_features"](w)) == ns["D"]


def test_window_features_constant_signal_is_finite(ns):
    w = np.tile([0.1, 0.2, -9.8, 0, 0, 0], (ns["WIN"], 1))
    assert np.isfinite(ns["window_features"](w)).all()


def _toy_model(ns):
    m = ns["GaussianHMM"](2, 1)
    m.pi = np.array([0.6, 0.4]); m.A = np.array([[0.7, 0.3], [0.2, 0.8]])
    m.mu = np.array([[0.0], [3.0]]); m.var = np.array([[1.0], [1.0]])
    return m


def test_forward_matches_brute_force(ns):
    m, X = _toy_model(ns), np.array([[0.2], [2.5], [3.1]])
    logB = m.log_emission(X); ll = m.forward(logB)[1]
    total = 0.0
    for z in itertools.product(range(2), repeat=len(X)):
        p = m.pi[z[0]] * np.exp(logB[0, z[0]])
        for t in range(1, len(X)): p *= m.A[z[t - 1], z[t]] * np.exp(logB[t, z[t]])
        total += p
    assert ll == pytest.approx(np.log(total))


def test_backward_gives_same_likelihood(ns):
    m, X = _toy_model(ns), np.array([[0.2], [2.5], [3.1], [0.1]])
    logB = m.log_emission(X); la, ll = m.forward(logB); lb = m.backward(logB)
    assert np.logaddexp.reduce(np.log(m.pi) + logB[0] + lb[0]) == pytest.approx(ll)
    gamma = np.exp(la + lb - ll)
    assert np.allclose(gamma.sum(1), 1)


def test_viterbi_matches_brute_force(ns):
    m, X = _toy_model(ns), np.array([[0.2], [2.5], [3.1], [0.1]])
    logB = m.log_emission(X); best, best_lp = None, -np.inf
    for z in itertools.product(range(2), repeat=len(X)):
        lp = np.log(m.pi[z[0]]) + logB[0, z[0]] + sum(np.log(m.A[z[t - 1], z[t]]) + logB[t, z[t]] for t in range(1, len(X)))
        if lp > best_lp: best, best_lp = z, lp
    path, lp = m.viterbi(X)
    assert tuple(path) == best and lp == pytest.approx(best_lp)


def test_supervised_fit_gives_valid_probabilities(ns):
    rng = np.random.default_rng(0)
    y = [np.array([0] * 20 + [1] * 20), np.array([1] * 15 + [0] * 15)]
    X = [rng.normal(3.0 * yy[:, None], 1.0) for yy in y]
    m = ns["GaussianHMM"](2, 1).fit_supervised(X, y)
    assert m.pi.sum() == pytest.approx(1) and np.allclose(m.A.sum(1), 1) and (m.A > 0).all()
    assert m.mu[1, 0] > m.mu[0, 0]


def test_baum_welch_converges_and_keeps_valid_matrices(ns):
    rng = np.random.default_rng(1)
    y = [np.array([0] * 25 + [1] * 25 + [0] * 25), np.array([1] * 30 + [0] * 20)]
    X = [rng.normal(3.0 * yy[:, None], 1.0) for yy in y]
    m = ns["GaussianHMM"](2, 1).fit_supervised(X, y).fit_baum_welch(X, max_iter=100, verbose=False)
    assert len(m.history) < 100                              # stopped by the tolerance, not the iteration limit
    assert np.all(np.diff(m.history) > -1e-2)                # log-likelihood does not drop (apart from tiny floor effects)
    assert m.pi.sum() == pytest.approx(1) and np.allclose(m.A.sum(1), 1)
    assert all((m.viterbi(x)[0] == yy).mean() > 0.9 for x, yy in zip(X, y))


class _FakeModel:
    def __init__(self, mode): self.mode = mode
    def viterbi(self, Z): return (Z[:, 0].astype(int) if self.mode == "perfect" else np.zeros(len(Z), int)), 0.0


def _sessions(ns):
    y = np.array([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)
    return [{"Z": y[:, None].astype(float), "y": y}]


def test_evaluate_perfect_predictions(ns):
    tab, cm, acc, _, _ = ns["evaluate"](_FakeModel("perfect"), _sessions(ns))
    assert acc == 1.0 and np.array_equal(cm, 4 * np.eye(4, dtype=int))
    assert (tab[["Sensitivity", "Specificity", "Precision", "F1", "Accuracy"]] == 1.0).all().all()


def test_evaluate_class_never_predicted_does_not_crash(ns):
    tab, cm, acc, _, _ = ns["evaluate"](_FakeModel("zeros"), _sessions(ns))
    assert acc == pytest.approx(0.25)
    assert tab.loc[1, "Sensitivity"] == 0 and tab.loc[1, "Precision"] == 0 and tab.loc[1, "F1"] == 0   # never predicted
    assert tab.loc[0, "Sensitivity"] == 1 and tab.loc[0, "Specificity"] == 0
