#!/usr/bin/env python3
import json; from pathlib import Path; import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
r = json.load(open("docs/research/general_ml/teacher_student_mnist.json"))
order = [("dense_N500_k1_sk0.0", "N=500"), ("dense_N2000_k1_sk0.0", "N=2000"), ("dense_N8000_k1_sk0.0", "N=8000"), ("key_N2000_k1_sk0.0", "N=2000\n+ unique key"), ("keynoise_N2000_k1_sk0.5", "key + noise\nσ=0.5"), ("keynoise_N2000_k1_sk1.0", "key + noise\nσ=1.0"), ("key_N2000_k8_sk0.0", "key, k=8\nper key")]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(len(order)); sp = [r[k]["curve"][-1]["spread_over_true"] for k, _ in order]; es = [r[k]["curve"][-1]["energy_score"] for k, _ in order]; esr = [r[k]["curve"][-1]["energy_score_reference"] for k, _ in order]
cols = ["tab:blue"] * 3 + ["tab:red"] + ["tab:orange"] * 2 + ["tab:green"]
ax[0].bar(x, sp, color=cols); ax[0].axhline(1, color="k", ls="--", lw=1); ax[0].set_xticks(x); ax[0].set_xticklabels([l for _, l in order], fontsize=8); ax[0].set_ylabel("spread / true posterior spread"); ax[0].set_title("MNIST teacher–student, 15k steps, 32 members")
ax[1].bar(x, es, color=cols, label="student"); ax[1].plot(x, esr, "k_", ms=18, mew=2, label="true posterior (reference)"); ax[1].set_xticks(x); ax[1].set_xticklabels([l for _, l in order], fontsize=8); ax[1].set_ylabel("energy score vs fresh true draw"); ax[1].legend(fontsize=8); ax[1].set_title("Density, identifying key, key noise")
fig.tight_layout(); fig.savefig("docs/research/figures/fig_mnist_teacher_student.png", dpi=120); print("saved")
