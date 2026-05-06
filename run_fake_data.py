from fake_data import generate_data

data = generate_data()

y_test     = data["y_test"]      # riktiga etiketter (heltal)
y_pred     = data["y_pred"]      # prediktioner (~92 % accuracy, med realistiska fel)
le         = data["le"]          # LabelEncoder → le.classes_ ger strängnamnen
best_k     = data["best_k"]     # bästa k-värdet (5)
accuracies = data["accuracies"] # accuracy per k (1–20), realistisk kurva
k_range    = data["k_range"]    # [1, 2, ..., 20]

