import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import label_binarize

def check_class_balance(y, le):
    """Analyserar om datasetet är obalanserat."""
    classes, counts = np.unique(y, return_counts=True)
    ratios = counts / counts.sum()
    if np.max(ratios) / np.min(ratios) > 2:
        print("\n[VARNING] Datasetet är obalanserat! Majoritetsklassen är mer än dubbelt så stor som minoritetsklassen.")
    else:
        print("\nDatasetet ser balanserat ut.")

def compute_metrics(y_test, y_pred, le):
    """Beräknar och returnerar utvärderingsmått inklusive PR-data."""
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=None)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    
    # Binarisera för PR-kurvor
    y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))
    # Simulerar sannolikheter för PR-kurvor baserat på 'y_pred'
    y_score = label_binarize(y_pred, classes=range(len(le.classes_))) 
    
    return cm, f1, report, y_test_bin, y_score

def interpret_results(cm, f1, report, data):
    """Analyserar och skriver ut en pedagogisk tolkning av resultaten."""
    print("\n--- Analys av modellresultat ---")
    print(f"Total Accuracy: {report['accuracy']:.2%}")
    print(f"Macro Avg F1-score: {report['macro avg']['f1-score']:.3f}")

    # Analysera F1-scores
    low_f1 = [cls for cls, score in zip(data['le'].classes_, f1) if score < 0.8]
    if low_f1:
        print(f"\nUppmärksamhet: Följande klasser har lägre prestanda (F1 < 0.8): {', '.join(low_f1[:5])}...")
        print("Tolkning: Modellen har svårt att skilja dessa klasser från andra. Överväg mer data för dessa tecken.")
    else:
        print("\nTolkning: Alla klasser presterar stabilt med F1-scores över 0.8.")

    # Analysera Confusion Matrix
    diag = np.diag(cm)
    row_sums = cm.sum(axis=1)
    errors = row_sums - diag
    max_error_idx = np.argmax(errors)
    print(f"\nKlass med flest förväxlingar: '{data['le'].classes_[max_error_idx]}' "
          f"({errors[max_error_idx]} felaktiga klassificeringar).")
    print("Tolkning: Denna klass är den största källan till osäkerhet i modellen.")
    
    print("\n--- Analys av Precision-Recall ---")
    print("Precision-Recall-kurvan visar avvägningen mellan hur många av våra positiva gissningar som är korrekta (Precision)")
    print("och hur många av de faktiska positiva fallen vi lyckades hitta (Recall).")

    print("\nSlutsats: Använd dessa insikter för att trimma 'k'-värdet eller undersöka om PCA-komponenterna behöver justeras.")
