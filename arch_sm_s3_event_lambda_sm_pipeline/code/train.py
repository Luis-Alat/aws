import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # Argumentos
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', None))

    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=None)

    args = parser.parse_args()

    # Cargar datos de entrenamiento
    train_data = pd.read_csv(os.path.join(args.train, 'train.csv'), header=None)
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
    model.fit(X_train, y_train)

    # Evaluar si hay datos de validación
    if args.validation and os.path.exists(os.path.join(args.validation, 'validation.csv')):
        val_data = pd.read_csv(os.path.join(args.validation, 'validation.csv'), header=None)
        X_val = val_data.iloc[:, :-1]
        y_val = val_data.iloc[:, -1]
        predictions = model.predict(X_val)
        acc = accuracy_score(y_val, predictions)
        print(f"Validation Accuracy: {acc:.4f}")

    # Guardar modelo
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

if __name__ == '__main__':
    main()