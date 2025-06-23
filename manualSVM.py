import os
import numpy as np
import random
from scipy.io import wavfile
from scipy.fftpack import dct

def extract_features(signal, sr):
    signal = signal / np.max(np.abs(signal))

    def windowed_features(sig):
        frame_size = int(0.025 * sr)
        frame_step = int(0.01 * sr)
        signal_len = len(sig)
        num_frames = int(np.ceil((signal_len - frame_size) / frame_step)) + 1

        pad_len = (num_frames - 1) * frame_step + frame_size
        pad_signal = np.append(sig, np.zeros(pad_len - signal_len))

        indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + \
                  np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_size, 1)).T
        frames = pad_signal[indices] * np.hamming(frame_size)

        NFFT = 512
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + sr / 2 / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, 28)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin = np.floor((NFFT + 1) * hz_points / sr)

        fbank = np.zeros((26, int(NFFT / 2 + 1)))
        for m in range(1, 27):
            f_m_minus, f_m, f_m_plus = int(bin[m - 1]), int(bin[m]), int(bin[m + 1])
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        log_fbank = np.log(filter_banks)

        mfccs = dct(log_fbank, type=2, axis=1, norm='ortho')[:, 1:21]
        delta = np.diff(mfccs, axis=0, prepend=mfccs[0:1])
        delta2 = np.diff(delta, axis=0, prepend=delta[0:1])

        features = np.concatenate([
            np.mean(mfccs, axis=0), np.std(mfccs, axis=0),
            np.mean(delta, axis=0), np.std(delta, axis=0),
            np.mean(delta2, axis=0), np.std(delta2, axis=0)
        ])
        return features

    energy = np.sum(signal ** 2)
    zero_crossings = ((signal[:-1] * signal[1:]) < 0).sum()
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1/sr)
    spectral_centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)

    base_feats = np.array([energy, zero_crossings, spectral_centroid])
    window_feats = windowed_features(signal)

    return np.concatenate((base_feats, window_feats))


def load_audio_dataset(directory='recordings'):
    X, y = [], []
    for fname in os.listdir(directory):
        if fname.endswith('.wav'):
            label = int(fname[0])
            sr, signal = wavfile.read(os.path.join(directory, fname))
            X.append(extract_features(signal, sr))
            y.append(label)
    return np.array(X), np.array(y)


def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


class LinearSVM:
    def __init__(self, num_features, lr=0.1, C=1, epochs=10):
        self.w = np.zeros(num_features)
        self.b = 0
        self.lr = lr
        self.C = C
        self.epochs = epochs

    def fit(self, X, y):
        for epoch in range(self.epochs):
            eta = self.lr / (1 + epoch)
            for i in range(len(X)):
                xi, yi = X[i], y[i]
                if yi * (np.dot(self.w, xi) + self.b) >= 1:
                    self.w -= eta * self.w
                else:
                    self.w -= eta * (self.w - self.C * yi * xi)
                    self.b += eta * self.C * yi

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)


class OneVSRestSVM:
    def __init__(self, num_classes, num_features, lr=0.1, C=1, epochs=10):
        self.models = [LinearSVM(num_features, lr, C, epochs) for _ in range(num_classes)]

    def fit(self, X, y):
        for i, model in enumerate(self.models):
            binary_y = np.where(y == i, 1, -1)
            model.fit(X, binary_y)

    def predict(self, X):
        scores = np.array([np.dot(X, model.w) + model.b for model in self.models])
        return np.argmax(scores, axis=0)


def evaluate_model(y_true, y_pred, num_classes, label_names):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        print(f"{label_names[i]} -> Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

    acc = np.mean(y_true == y_pred)
    print(f"\nOverall Accuracy: {acc * 100:.2f}%")
    return cm
    

def k_fold_split(X, y, k=5, seed=42):
    data = list(zip(X, y))
    random.Random(seed).shuffle(data)
    fold_size = len(data) // k
    folds = []

    for i in range(k):
        test_data = data[i*fold_size:(i+1)*fold_size]
        train_data = data[:i*fold_size] + data[(i+1)*fold_size:]
        X_train, y_train = zip(*train_data)
        X_test, y_test = zip(*test_data)
        folds.append((np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)))

    return folds


def cross_validate(X, y, num_classes, num_features, k=5, lr=0.01, C=1.0, epochs=50):
    folds = k_fold_split(X, y, k)
    acc_scores = []

    for fold_idx, (X_train, y_train, X_val, y_val) in enumerate(folds):
        print(f"Fold {fold_idx+1}/{k}")
        model = OneVSRestSVM(num_classes, num_features, lr=lr, C=C, epochs=epochs)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = np.mean(preds == y_val)
        acc_scores.append(acc)
        print(f"Accuracy: {acc * 100:.2f}%")

    return np.mean(acc_scores)


def grid_search(X, y, num_classes, num_features, k=5):
    lr_values = [0.001, 0.01, 0.1]
    C_values = [0.05, 0.5, 1]
    epoch_values = [50, 100]

    best_score = 0
    best_params = {}

    for lr in lr_values:
        for C in C_values:
            for epochs in epoch_values:
                print(f"\nTrying params: lr={lr}, C={C}, epochs={epochs}")
                score = cross_validate(X, y, num_classes, num_features, k, lr, C, epochs)
                print(f"Average Accuracy: {score * 100:.2f}%")

                if score > best_score:
                    best_score = score
                    best_params = {'lr': lr, 'C': C, 'epochs': epochs}

    print(f"\nBest Params: {best_params}, Best Accuracy: {best_score * 100:.2f}%")
    return best_params


def evaluate_model(y_true, y_pred, num_classes, label_names):
    cm = confusion_matrix_np(y_true, y_pred, num_classes)

    precision = []
    recall = []
    f1 = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)

        print(f"{label_names[i]} -> Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1_score:.2f}")

    overall_acc = np.mean(np.array(y_pred) == np.array(y_true))
    print(f"\nOverall Accuracy: {overall_acc * 100:.2f}%")
    
    return cm, precision, recall, f1