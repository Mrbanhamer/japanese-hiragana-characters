import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── Konfiguration ────────────────────────────────
KLASSER = [
    "aa", "chi", "ee", "fu", "ha", "he", "hi", "ho", "ii",
    "ka", "ke", "ki", "ko", "ku", "ma", "me", "mi", "mo", "mu",
    "na", "ne", "ni", "nn", "no", "nu", "oo", "ra", "re", "ri",
    "ro", "ru", "sa", "se", "shi", "so", "su", "ta", "te", "to",
    "tsu", "uu", "wa", "wo", "ya", "yo", "yu",
]
N_PER_KLASS = 100
TEST_RATIO = 0.2
N_COMPONENTS = 170
K_RANGE = range(1, 21)
RANDOM_STATE = 42

def generate_data():
    """Returnerar data i samma format som den riktiga pipelinen."""
    rng = np.random.default_rng(RANDOM_STATE)
    le = LabelEncoder()
    le.fit(KLASSER)

    n_total = N_PER_KLASS * len(KLASSER)
    n_test = int(n_total * TEST_RATIO)
    n_train = n_total - n_test

    y_all = np.repeat(np.arange(len(KLASSER)), N_PER_KLASS)
    rng.shuffle(y_all)
    y_train = y_all[:n_train]
    y_test = y_all[n_train:]

    centroids = rng.standard_normal((len(KLASSER), N_COMPONENTS))

    def make_features(labels):
        features = np.zeros((len(labels), N_COMPONENTS))
        for i, label in enumerate(labels):
            features[i] = centroids[label] + rng.standard_normal(N_COMPONENTS) * 0.3
        return features

    x_train = make_features(y_train)
    x_test = make_features(y_test)

    y_pred = y_test.copy()
    n_errors = int(len(y_test) * 0.08)
    error_idx = rng.choice(len(y_test), size=n_errors, replace=False)
    for idx in error_idx:
        wrong = rng.choice([k for k in range(len(KLASSER)) if k != y_test[idx]])
        y_pred[idx] = wrong

    accuracies = []
    for k in K_RANGE:
        base = 0.88 + 0.04 * np.exp(-0.5 * ((k - 5) / 2) ** 2)
        noise = rng.uniform(-0.005, 0.005)
        accuracies.append(min(base + noise, 0.99))

    best_k = int(K_RANGE[np.argmax(accuracies)])

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "le": le,
        "best_k": best_k,
        "accuracies": accuracies,
        "k_range": list(K_RANGE),
    }
