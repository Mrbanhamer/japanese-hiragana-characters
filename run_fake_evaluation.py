from fake_data import generate_data
from evaluation import check_class_balance, compute_metrics, interpret_results
from plotting import plot_results

def run():
    print("Genererar fejkdata...")
    data = generate_data()
    
    check_class_balance(data['y_test'], data['le'])
    
    print("Beräknar mått...")
    cm, f1, report, y_test_bin, y_score = compute_metrics(data['y_test'], data['y_pred'], data['le'])
    
    interpret_results(cm, f1, report, data)
    
    print("Visualiserar...")
    plot_results(data, cm, f1, report, y_test_bin, y_score)

if __name__ == "__main__":
    run()
