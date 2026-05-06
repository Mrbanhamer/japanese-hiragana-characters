from process_image.image_liquditation import run_dataset_processing
from data_processing import pca_reduction
from model_training import train

if __name__ == "__main__":

    x_raw, y_raw = run_dataset_processing()

    x_train_pca, x_test_pca, y_train, y_test, le = pca_reduction(x_raw, y_raw)

    y_pred, best_k, accuracies = train(x_train_pca, x_test_pca, y_train, y_test)