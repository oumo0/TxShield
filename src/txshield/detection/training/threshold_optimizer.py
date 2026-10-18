import numpy as np
from sklearn.metrics import f1_score

def find_best_f1_threshold(y_true, probabilities) -> tuple[float, float]:
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        score = f1_score(y_true, probabilities >= threshold, zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score
