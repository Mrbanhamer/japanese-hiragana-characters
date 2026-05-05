from data_processing import pca_reduction
from model_training import train
from main import check_class_balance, compute_metrics, interpret_results, plot_results

def run(X_raw, y_raw):
    # Förbehandling & PCA
    print("Kör PCA-reduktion...")
    x_train_pca, x_test_pca, y_train, y_test, le = pca_reduction(X_raw, y_raw)
    
    # Modellträning & Tuning
    print("Tränar och utvärderar KNN...")
    y_pred, best_k, accuracies = train(x_train_pca, x_test_pca, y_train, y_test)
    
    # Utvärdering
    data = {
        "x_test": x_test_pca,
        "y_test": y_test,
        "y_pred": y_pred,
        "le": le,
        "best_k": best_k,
        "accuracies": accuracies
    }
    
    check_class_balance(y_test, le)
    
    cm, f1, report, y_test_bin, y_score = compute_metrics(y_test, y_pred, le)
    
    interpret_results(cm, f1, report, data)
    
    print("Visualiserar...")
    plot_results(data, cm, f1, report, y_test_bin, y_score)

# Exempel på hur man kör den när bilderna är inlästa:
# if __name__ == "__main__":
#     # X_raw, y_raw = load_images()  <-- Denna funktion behöver vi skapa när vi har bilderna
#     # run(X_raw, y_raw)
