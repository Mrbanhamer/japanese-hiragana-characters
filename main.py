from process_image.image_liquditation import run_dataset_processing
from data_processing import pca_reduction
from model_training import train
from evaluation import compute_metrics, interpret_results
from plotting import plot_results
from settings import K_RANGE

if __name__ == "__main__":

    x_raw, y_raw = run_dataset_processing()

    x_train_pca, x_test_pca, y_train, y_test, le = pca_reduction(x_raw, y_raw)

    y_pred, best_k, accuracies = train(x_train_pca, x_test_pca, y_train, y_test)

    cm, f1, report, y_test_bin, y_score = compute_metrics(y_test, y_pred, le)

    data = {
            "x_test": x_test_pca,
            "y_test": y_test,
            "y_pred": y_pred,
            "le": le,
            "best_k": best_k,
            "accuracies": accuracies,
            "k_range": K_RANGE
        }

    cm, f1, report, y_test_bin, y_score = compute_metrics(y_test, y_pred, le)

    interpret_results(cm, f1, report, data)

    plot_results(data, cm, f1, report, y_test_bin, y_score)