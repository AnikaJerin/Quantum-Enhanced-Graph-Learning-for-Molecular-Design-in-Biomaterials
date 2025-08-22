from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC

def rf_regressor(): return RandomForestRegressor(n_estimators=300, random_state=42)
def rf_classifier(): return RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)

def krr(alpha=1e-2): return KernelRidge(alpha=alpha, kernel="precomputed")  # supply K matrix

def svc_from_precomputed(C=1.0, class_weight=None, probability=True):
    return SVC(C=C, kernel="precomputed", class_weight=class_weight, probability=probability, random_state=42)
