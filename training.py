import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from manualSVM import (
    load_audio_dataset,
    OneVSRestSVM,
    evaluate_model,
    grid_search
)

# --------------------------------------
# CONFIG
# --------------------------------------

DATASET_DIR = 'recordings'
MODEL_FILE = 'model/svm_speech_model.pkl'
SEED = 42
TRAIN_RATIO = 0.8

# --------------------------------------
# UTILITIES
# --------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def normalize_features(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, mean, std

def split_dataset(X, y, train_ratio=0.8, seed=42):
    data = list(zip(X, y))
    random.Random(seed).shuffle(data)
    split_idx = int(train_ratio * len(data))
    train, test = data[:split_idx], data[split_idx:]
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# --------------------------------------
# MAIN TRAINING PIPELINE
# --------------------------------------

def main():
    set_seed(SEED)

    # Load and preprocess dataset
    X, y = load_audio_dataset(DATASET_DIR)
    X_train, y_train, X_test, y_test = split_dataset(X, y, TRAIN_RATIO, SEED)
    X_train, X_test, mean, std = normalize_features(X_train, X_test)

    num_classes = len(set(y))
    num_features = X.shape[1]
    label_names = [str(i) for i in sorted(set(y))]

    # Hyperparameter tuning
    best_params = grid_search(
        X_train, y_train,
        num_classes, num_features,
        k=5
    )

    # Final training
    model = OneVSRestSVM(
        num_classes,
        num_features,
        lr=best_params['lr'],
        C=best_params['C'],
        epochs=best_params['epochs']
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    cm, precision, recall, f1 = evaluate_model(y_test, y_pred, num_classes, label_names)

    accuracy = np.mean(y_pred == y_test)
    precision_avg = np.mean(precision)
    recall_avg = np.mean(recall)
    f1_avg = np.mean(f1)

    plot_confusion_matrix(cm, label_names)

    # Save model
    model_package = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision_avg,
        'recall': recall_avg,
        'f1': f1_avg,
        'label_names': label_names,
        'mean': mean,
        'std': std
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_package, f)

    print(f"✅ Model saved to {MODEL_FILE}")
    print(f"🎯 Accuracy: {accuracy:.2%}, Precision: {precision_avg:.2%}, Recall: {recall_avg:.2%}, F1: {f1_avg:.2%}")

# --------------------------------------

if __name__ == '__main__':
    main()