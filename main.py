import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from fake_data import generate

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

def plot_results(data, cm, f1, report, y_test_bin, y_score):
    """Visualiserar resultat i separata fönster för maximal läsbarhet."""
    accuracies = data['accuracies']
    k_range = data['k_range']
    le = data['le']

    # Fönster 1: Prestanda (Kurva och Sammanfattning)
    fig1 = plt.figure(figsize=(12, 5))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax1.plot(k_range, accuracies, marker='o', linestyle='-', color='b')
    ax1.set_title("k-accuracy-kurva")
    ax1.grid(True)

    ax2 = fig1.add_subplot(1, 2, 2)
    summary_text = (f"Total Accuracy: {report['accuracy']:.3f}\n"
                    f"Macro Avg F1: {report['macro avg']['f1-score']:.3f}\n"
                    f"Best k: {data['best_k']}")
    ax2.text(0.1, 0.5, summary_text, fontsize=14, bbox=dict(facecolor='white', alpha=0.5))
    ax2.set_title("Sammanfattning")
    ax2.axis('off')
    plt.tight_layout()

    # Fönster 2: F1-score
    fig2 = plt.figure(figsize=(16, 6))
    ax3 = fig2.add_subplot(1, 1, 1)
    ax3.bar(le.classes_, f1, color='teal')
    ax3.set_title("F1-score per klass")
    ax3.tick_params(axis='x', rotation=90)
    ax3.grid(True)
    plt.tight_layout()

    # Fönster 3: Förväxlingsmatriser
    fig3 = plt.figure(figsize=(16, 8))
    ax4 = fig3.add_subplot(1, 2, 1)
    sns.heatmap(cm, annot=False, cmap='Blues', ax=ax4)
    ax4.set_title("Förväxlingsmatris (full)")

    diag = np.diag(cm)
    row_sums = cm.sum(axis=1)
    errors = row_sums - diag
    top_err_idx = np.argsort(errors)[-5:]
    cm_focused = cm[np.ix_(top_err_idx, top_err_idx)]
    labels_focused = le.classes_[top_err_idx]
    
    ax5 = fig3.add_subplot(1, 2, 2)
    sns.heatmap(cm_focused, annot=True, fmt='d', cmap='Reds', ax=ax5, 
                xticklabels=labels_focused, yticklabels=labels_focused)
    ax5.set_title("Förväxlingsmatris (topp 5 fel)")
    plt.tight_layout()

    # Fönster 4: Precision-Recall (topp 5 klasser)
    fig4 = plt.figure(figsize=(10, 8))
    ax6 = fig4.add_subplot(1, 1, 1)
    n_classes = y_test_bin.shape[1]
    for i in range(min(5, n_classes)):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
        ax6.plot(recall, precision, label=f'Klass {le.classes_[i]} (AP={ap:.2f})')
    ax6.set_title("Precision-Recall-kurvor (topp 5 klasser)")
    ax6.set_xlabel("Recall")
    ax6.set_ylabel("Precision")
    ax6.legend()
    ax6.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Genererar fejkdata...")
    data = generate()
    
    check_class_balance(data['y_test'], data['le'])
    
    print("Beräknar mått...")
    cm, f1, report, y_test_bin, y_score = compute_metrics(data['y_test'], data['y_pred'], data['le'])
    
    interpret_results(cm, f1, report, data)
    
    print("Visualiserar...")
    plot_results(data, cm, f1, report, y_test_bin, y_score)
