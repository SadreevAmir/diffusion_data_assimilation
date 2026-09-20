"""Exact finite-support check of the correction discussed in the review.
This script does NOT reproduce the research agent's training or results.
Uses k(x,y)=1[x=y] for Bernoulli forecasts; the score is squared
Euclidean distance between probability vectors and the realized one-hot target.
"""
import json
import math
from pathlib import Path


def convolution(a, b):
    out = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i+j] += ai * bj
    return out


def conditional_binomial(n, p):
    return [math.comb(n,k) * p**k * (1-p)**(n-k) for k in range(n+1)]


# Independent fair key, Bernoulli p=0.1 or 0.9 conditional on key.
# Independent truth Bernoulli(0.5).
V, W, Rbar, M = 0.32, 0.18, 0.5, 64
exact_results=[]
for L in [1,2,4,8,16,64]:
    m=M//L
    p0,p1=conditional_binomial(m,.1),conditional_binomial(m,.9)
    group=[.5*(a+b) for a,b in zip(p0,p1)]
    total=[1.]
    for _ in range(L):
        total=convolution(total,group)
    # E_Y ||(1-q,q) - onehot(Y)||^2 = q^2+(1-q)^2
    score=sum(prob*((j/M)**2+(1-j/M)**2) for j,prob in enumerate(total))
    prediction=Rbar+V/L+W/M
    assert abs(score-prediction)<1e-12,(L,score,prediction)
    exact_results.append({'M':M,'L':L,'exact_score':score,'prediction':prediction})

# New key per member: its Monte Carlo penalty is (V+W)/B, not W/B.
B=4096
bad=Rbar+(V+W)/B-W/B
correct=Rbar+(V+W)/B-(V+W)/B
assert abs(bad-Rbar-V/B)<1e-14
assert correct==Rbar

reported=[('K0',.0401,.0001),('Kfresh',.0316,.0010),
          ('Krho05',.0384,.0029),('Krho09',.0582,.0148),('K1',.0836,.0387)]
plug_in=[{'arm':arm,'reported_raw_MMD2':raw,'reported_V':v,
          'predicted_mixture_MMD2':round(raw-v,7)} for arm,raw,v in reported]
# Valid only under matching score/kernel normalization across P1 and T3.
result={'verification':'exact enumeration, not a new agent run',
        'grouped_forecast_checks':exact_results,
        'mixture_bank_B':B,'incorrect_correction_bias':bad-Rbar,
        'corrected_risk':correct,'report_plugin_calculations':plug_in}
path=Path(__file__).with_name('algebra_checks.json')
path.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
