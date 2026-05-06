# Huvudpipeline: ladda rådata → PCA-reducera → träna KNN → utvärdera → visualisera
from process_image.image_liquditation import run_dataset_processing
from data_processing import pca_reduction
from model_training import train
from evaluation import compute_metrics, interpret_results
from plotting import plot_results
from settings import K_RANGE

if __name__ == "__main__":

    # Steg 1 – Ladda datasetet: varje bild blir en flat numpy-array (x) med tillhörande etikett (y)
    x_raw, y_raw = run_dataset_processing()

    # Steg 2 – Dela i träning/test och reducera dimensionerna med PCA.
    # LabelEncoder (le) behövs senare för att mappa numeriska etiketter tillbaka till teckennamn.
    x_train_pca, x_test_pca, y_train, y_test, le = pca_reduction(x_raw, y_raw)

    # Steg 3 – Träna KNN: tune_k provar k=1..20, sedan tränas en slutgiltig modell med bästa k.
    # Returnerar prediktioner (y_pred) samt accuracy per k (för k-kurvan i plottningen).
    y_pred, best_k, accuracies = train(x_train_pca, x_test_pca, y_train, y_test)

    # Samla allt som plottnings- och tolkningssteget behöver i ett gemensamt dict
    data = {
            "x_test": x_test_pca,
            "y_test": y_test,
            "y_pred": y_pred,
            "le": le,
            "best_k": best_k,
            "accuracies": accuracies,
            "k_range": K_RANGE
        }

    # Steg 4 – Beräkna utvärderingsmått: confusion matrix, F1-score, klassificeringsrapport
    # samt binariserad y_test/y_score (behövs för ROC-kurvan)
    cm, f1, report, y_test_bin, y_score = compute_metrics(y_test, y_pred, le)

    # Steg 5 – Skriv ut en texttolkning av resultaten (accuracy, F1, svaga/starka klasser)
    interpret_results(cm, f1, report, data)

    # Steg 6 – Generera alla diagram: k-kurva, confusion matrix, ROC, m.m.
    plot_results(data, cm, f1, report, y_test_bin, y_score)