"""
Module: svm_classifier

Description:
This module implements the training of the Support Vector Machine (SVM)
classifier used for presentation attack detection (PAD).

The classifier is trained on the feature vectors extracted from face
images and learns to distinguish between live samples and spoof
attacks. Feature normalization is performed using StandardScaler
before training in order to ensure that all features contribute
equally to the decision boundary.

The trained classifier and the fitted scaler are returned and will
later be used for PAD inference during runtime.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def train_pad_classifier(X_train, y_train):
    """
    Train a PAD classifier on the provided training data.

    Parameters:
    - X_train: A 2D numpy array (n_samples, n_features) containing the feature vectors.
    - y_train: A 1D numpy array (n_samples,) containing the class labels.

    Returns:
    - scaler: The StandardScaler used to normalize the features.
    - classifier: The trained SVM classifier.
    """

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train the SVM classifier
    classifier = SVC(kernel='rbf', class_weight="balanced", probability=True)
    classifier.fit(X_train_scaled, y_train)

    return scaler, classifier