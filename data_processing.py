from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# X_raw : array -> Bilder
# y_raw : list -> Strängetiketter
def split_data(X_raw, y_raw):
    """Stratified train/test split.

    Stratifiering är viktigt här eftersom vissa tecken kan likna
    varandra mer än andra — vi vill att varje klass har samma
    andel i båda uppsättningarna så att accuracy inte snedvrids.
    """
    y, le = encode_labels(y_raw)

    print("Delar upp datan i stratifierade delar.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=42,
    )

    print(f"  Träning: {len(X_train)}  Test: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, le
