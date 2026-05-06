import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from settings import K_RANGE


# Vi väljer K optimistisk utifrån testdatan, utan att augmentera/maskera någon data. För ett mer obalanserat dataset bör cross validation implementeras.
def tune_k(x_train_pca, x_test_pca, y_train, y_test):
    """Provar varje k i K_RANGE och returnerar det bästa.

    För varje k anpassar vi en ny KNN, predikterar på testdatan och
    loggar accuracy. Det k som ger högst accuracy vinner. Vid
    lika resultat väljer numpy.argmax det första (= minsta k),
    vilket är bra eftersom ett mindre k tenderar att fånga finare
    detaljer.

    Returnerar
    ----------
    best_k     : int         — det vinnande k-värdet
    accuracies : list[float] — accuracy för varje k (samma ordning som K_RANGE),
                               sparas så vi kan plotta k-mot-accuracy-kurvan
    """

    accuracies = []

    for k in K_RANGE:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train_pca, y_train)

        acc = accuracy_score(y_test, knn.predict(x_test_pca))
        accuracies.append(acc)

        print(f"k={k:>2d} accuracy={acc:.4f}")

    best_k = K_RANGE[np.argmax(accuracies)]
    print(f"\nBästa k: {best_k} ({max(accuracies):.4f})")

    return best_k, accuracies

def train(x_train_pca, x_test_pca, y_train, y_test):
    """Tränar den slutgiltiga modellen med bästa k och returnerar prediktioner.

    Vi anropar tune_k först för att hitta best_k och tränar sedan
    en ny modell med det värdet. Det innebär att vi tränar k+1
    modeller totalt, det är inget problem för ett mindre dataset
    och låter oss hålla tuning-logiken prydligt separerad.

    Returnerar
    ----------
    y_pred     : np.ndarray  — predikterade etiketter för testdatan
    best_k     : int         — valt k (skickas vidare för plottning)
    accuracies : list[float] — accuracy per k (skickas vidare för plottning)
    """

    best_k, accuracies = tune_k(x_train_pca, x_test_pca, y_train, y_test)

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(x_train_pca, y_train)
    print(f"Modell tränad med k={best_k} på {len(x_train_pca)} bilder.")

    y_pred = model.predict(x_test_pca)
    print(f"Prediktion klar — {len(y_pred)} testbilder klassificerade.")
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2%}")

    return y_pred, best_k, accuracies