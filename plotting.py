import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score

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
