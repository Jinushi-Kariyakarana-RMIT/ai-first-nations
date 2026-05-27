# Ive mostly just taken Jakes work in 'mangroves.ipynb' and added it to one long .py file and changed stuff so it works with flask

import numpy as np
import tifffile
import joblib
# import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
# from io import BytesIO

zero_value = 0.0
seed = 42
image_shape = (256, 256, 7)

# directory = Path('PhDMangroveDataset')
# mangrove_class = ['mangroves', 'nonmangroves']
# labbeled_class = {'mangroves': 1, 'nonmangroves': 0}

# Base directory to find the model if saved
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / 'mangrove_model.joblib'

# The website currently trains a new model on startup if it cant find a saved model ('mangrove_model.joblib'), so the dataset should be in the same directory as this file for it to work.
# In production, we would want to train the model separately and have it saved, so the dataset could be stored elsewhere and not included in the deployment.
# To retrain the model, delete the 'mangrove_model.joblib' file and restart the Flask app. It will look for the dataset, train a new model, and save it as 'mangrove_model.joblib' for future use.

# Training data location and class labels
directory = BASE_DIR.parent / 'PhDMangroveDataset'
mangrove_class = ['Mangroves', 'NonMangroves']
labbeled_class = {'Mangroves': 1, 'NonMangroves': 0}

DATASET_PATH = directory
MANGROVE_CLASS = mangrove_class
LABELED_CLASS = labbeled_class


# for cls in mangrove_class:
#     folder = directory / cls

def corruption_check(path: str) -> str:
    try:
        #this is reading the file using tifffile (originally i used the OS method after i couldnt get tifffile to work however i reversed engineered someones kaggle code on this dataset to learn)
        data = tifffile.imread(path).astype(np.float32)
        if data.ndim == 2:
            data = data[..., np.newaxis] #this is adding a new axis so it goes from 2d to 3d + colour
        elif data.shape[0] < data.shape[-1]:
            data = data.transpose(1, 2, 0) #transposing the data so it goes from (bands, height, width) to (height, width, bands) 256x256x7
        valid = data[data != zero_value]
        if valid.size == 0:
            return 'dark'
        mean = float(valid.mean())
        std = float(valid.std())
        if mean < 0.01:
            return 'dark'
        if std < 0.001:
            return 'uniform'
        return 'valid'
    except Exception as e:
        return f'error: {e}'


def extract_statistical_features(band, no_data=0.0): #this is extracting statistical features from the image band such as mean, std, min, max, percentiles and IQR
    valid = band[band != no_data]
    if valid.size == 0:
        return [0.0] * 8
    p25, median, p75 = np.percentile(valid, [25, 50, 75])
    return [
        float(valid.mean()), float(valid.std()), float(valid.min()),
        float(valid.max()), p25, median, p75, float(p75 - p25)
    ]


def extract_glcm_features(band, no_data=0.0, n_levels=32): #this is extracting texture features GLCM (Gray Level Co-occurrence Matrix) features from the image band
    valid_mask = band != no_data
    if valid_mask.sum() == 0:
        return [0.0] * 6
    b_min, b_max = band[valid_mask].min(), band[valid_mask].max() #
    if b_max == b_min:
        return [0.0] * 6
    normalized = np.zeros_like(band, dtype=np.uint8)
    normalized[valid_mask] = (
        (band[valid_mask] - b_min) / (b_max - b_min) * (n_levels - 1)
    ).astype(np.uint8)
    glcm = graycomatrix(
        normalized, distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=n_levels, symmetric=True, normed=True
    ) #calcuating the GLCM matrix for the normalized band with specified distances and angles, levels, symmetric and normed parameters
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']  #these are the 6 GLCM properties we are extracting
    return [float(graycoprops(glcm, p).mean()) for p in props] #this is calculating the mean of each GLCM property


def extract_features_from_path(tif_path: str):
    try:
        data = tifffile.imread(tif_path).astype(np.float32) #same reading  annotation as described before
        if data.ndim == 2:
            data = data[..., np.newaxis]
        elif data.shape[0] < data.shape[-1]:
            data = data.transpose(1, 2, 0)
        features = []
        for c in range(data.shape[-1]):
            band = data[:, :, c]
            features.extend(extract_statistical_features(band, zero_value)) #this is extracting the statistical features for each band and adding them to the features list
            features.extend(extract_glcm_features(band, zero_value))  #this is extracting the GLCM features for each band and adding them to the features list
        return np.array(features, dtype=np.float32) #this is returning the features as a numpy array of type float32
    except Exception as e:
        print(f'Could not read {tif_path}: {e}')
        return None


def extract_features_from_pil_image(image, bands=None):
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            data = np.array(image).astype(np.float32)
        else:
            data = image.astype(np.float32)

        # Handle different image formats
        if data.ndim == 2:
            data = data[..., np.newaxis]
        elif data.ndim == 3 and data.shape[0] < data.shape[-1]:
            data = data.transpose(1, 2, 0)

        # Extract features from available bands
        features = []
        for c in range(min(data.shape[-1], bands or 7)):
            band = data[:, :, c]
            features.extend(extract_statistical_features(band, zero_value)) #this is extracting the statistical features for each band and adding them to the features list
            features.extend(extract_glcm_features(band, zero_value))  #this is extracting the GLCM features for each band and adding them to the features list

        return np.array(features, dtype=np.float32) #this is returning the features as a numpy array of type float32
    except Exception as e:
        print(f'Could not read uploaded image: {e}')
        return None


#this section of code is essentially going through every image and deciding which image is valido or not in mangroves and nonmangroves and printing the valid and invalid of each class
def load_training_data(max_samples=None):
    print('Determining which images are non valid')

    # valid_paths, valid_labels = [], [] #this is creating empty lists to store the paths and labels of the valid images
    # removed = {cls: 0 for cls in mangrove_class}
    # for path, label in zip(valid_paths, valid_labels):
    #     tag = corruption(path) #determing if the file is valid or not
    #     cls = [k for k, v in labbeled_class.items() if v == label][0] #this is finding the class name based on the label (1 for mangroves and 0 for non-mangroves)
    #     if tag == 'valid':
    #         valid_paths.append(path)
    #         valid_labels.append(label)
    #     else:
    #         removed[cls] += 1

    if not DATASET_PATH.exists():
        print(f"Error: Dataset path does not exist: {DATASET_PATH}")
        print(f"Expected at: {DATASET_PATH.absolute()}")
        return np.array([]), np.array([])

    print('Filtering corrupted images')
    valid_paths, valid_labels = [], []
    removed = {cls: 0 for cls in mangrove_class}
    for cls in mangrove_class:
        folder = directory / cls
        if not folder.exists():
            print(f"Warning: {folder} does not exist")
            continue

        tifs = sorted(folder.glob('*.tif'))
        for tif in tifs:
            if max_samples and len(valid_paths) >= max_samples:
                break
            tag = corruption_check(str(tif))
            if tag == 'valid':
                valid_paths.append(str(tif))
                valid_labels.append(labbeled_class[cls])
            else:
                removed[cls] += 1

    valid_paths = np.array(valid_paths)
    all_labels = np.array(valid_labels, dtype=np.int32)

    print(f'\nRemoved:')
    for cls, count in removed.items():
        print(f'  {cls}: {count} corrupt images removed')
    total = len(all_labels)
    n_pos = int(all_labels.sum())
    n_neg = total - n_pos
    # ratio = n_pos / n_neg if n_neg else float('inf')
    if total > 0:
        print(f'Mangrove  (1) : {n_pos:,}  ({100 * n_pos / total:.1f} %)')
        print(f'Non-mang. (0) : {n_neg:,}  ({100 * n_neg / total:.1f} %)')
    else:
        print('No valid images found')

    return valid_paths, all_labels


def loading_of_features(paths, labels, desc='', expected_len=None): #this is loading the features from the paths and labels and is also padding the features to the same length if expected_len is provided
    X, y, lengths = [], [], []
    for i, (path, label) in enumerate(zip(paths, labels)):
        if i % 100 == 0:
            print(f'  {desc}: {i}/{len(paths)}', end='\r')
        feats = extract_features_from_path(path)
        if feats is not None:
            X.append(feats)
            y.append(label)
            lengths.append(len(feats))
    print(f'  {desc}: {len(X)}/{len(paths)} loaded')
    print(f'  Feature lengths seen: {sorted(set(lengths))}' if lengths else f'  Feature lengths seen: []')

    if not X:
        return None, None, 0

    #Pad all vectors to the same length
    max_len = expected_len or max(lengths) #determing maximum lenght of featrures to pad to
    X_padded = np.zeros((len(X), max_len), dtype=np.float32)
    for i, feats in enumerate(X):
        X_padded[i, :len(feats)] = feats

    return X_padded, np.array(y), max_len


def train_model(save=True):
    print("Loading training data...")
    valid_paths, all_labels = load_training_data()

    if len(valid_paths) == 0:
        print("Error: No valid training data found!")
        return None

    # Splitting the data insto training, validation and testing using stratified, lastly the split is 70% training, 15% validation and 15% testing
    X_train_paths, X_tmp, y_train, y_tmp = train_test_split(
        valid_paths, all_labels, test_size=0.30, stratify=all_labels, random_state=seed
    )
    X_val_paths, X_test_paths, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed
    )

    print("Loading features...")
    X_train, y_train, max_len = loading_of_features(X_train_paths, y_train, 'Train')
    X_val,   y_val,   _       = loading_of_features(X_val_paths,   y_val,   'Val  ', max_len)
    X_test,  y_test,  _       = loading_of_features(X_test_paths,  y_test,  'Test ', max_len)

    if X_train is None or X_val is None or X_test is None:
        print("Error: Could not load features!")
        return None

    #validation set merge with the training set
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    #randomforest model pipline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1,
            verbose=1))
    ])
    print('Fitting Random Forest')
    pipeline.fit(X_train_full, y_train_full)
    y_pred       = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print(f'Test accuracy : {test_acc:.4f}')
    print(f'Test AUC      : {test_auc:.4f}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=['Non-Mangrove', 'Mangrove'], digits=4))

    if save:
        joblib.dump(pipeline, str(MODEL_PATH))
        print(f"Model saved to {MODEL_PATH}")

    return pipeline


def load_or_train_model():
    if MODEL_PATH.exists():
        print(f"Loading model from {MODEL_PATH}")
        return joblib.load(str(MODEL_PATH))
    else:
        print(f"Model not found at {MODEL_PATH}, training new model...")
        return train_model(save=True)


def predict_mangrove(image_path_or_array, model=None):
    if model is None:
        model = load_or_train_model()

    if model is None:
        return None, None, "Error: Model could not be loaded or trained"

    # Extract features
    if isinstance(image_path_or_array, str):
        features = extract_features_from_path(image_path_or_array)
    else:
        features = extract_features_from_pil_image(image_path_or_array)

    if features is None:
        return None, None, "Error: Could not extract features from image"

    # Pad features to match training
    try:
        model_features = model.named_steps['scaler'].mean_.shape[0]
        if len(features) < model_features:
            padded = np.zeros(model_features, dtype=np.float32)
            padded[:len(features)] = features
            features = padded
        else:
            features = features[:model_features]
    except Exception as e:
        print(f"Warning: Could not pad features: {e}")

    # Predict
    try:
        features_2d = features.reshape(1, -1)
        prediction = model.predict(features_2d)[0]
        confidence = model.predict_proba(features_2d)[0]

        result = {
            'prediction': 'Mangrove Detected' if prediction == 1 else 'Non-Mangrove',
            'confidence': float(confidence[prediction]),
            'mangrove_probability': float(confidence[1]),
            'non_mangrove_probability': float(confidence[0])
        }
        return result, features, None
    except Exception as e:
        return None, None, f"Error during prediction: {str(e)}"