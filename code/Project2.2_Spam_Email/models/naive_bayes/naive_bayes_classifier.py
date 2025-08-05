import numpy as np
import pickle
import os
from collections import Counter, defaultdict
from math import log, pi
from utils.preprocess import preprocess_text
from utils.logger import setup_logger
import time
from sklearn.feature_extraction.text import TfidfVectorizer

logger = setup_logger("naive_bayes")

def _create_default_dict_int():
    """Helper function to create defaultdict(int) for pickling compatibility."""
    return defaultdict(int)

class BaseNaiveBayes:
    """Base class for Naive Bayes classifiers with common functionality."""

    def __init__(self, vector_store, alpha=1.0):
        self.vector_store = vector_store
        self.alpha = alpha
        self.is_trained = False

        if vector_store is not None:
            self.messages = vector_store.train_metadata["Message"].tolist()
            self.labels = vector_store.train_metadata["Category"].tolist()
            self.num_docs = len(self.messages)

            self.class_counts = Counter()
            self.class_priors = {}

            self._train()
            self.is_trained = True

    def save_model(self, model_path):
        """Save the trained model to a file."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

        logger.info("Model saved to %s", model_path)

    @classmethod
    def load_model(cls, model_path):
        """Load a trained model from a file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.info("Model loaded from %s", model_path)
        return model

class MultinomialNaiveBayes(BaseNaiveBayes):
    """
    Multinomial Naive Bayes classifier for spam/ham email classification.
    Uses word frequency features with Laplace smoothing.
    """

    def __init__(self, vector_store=None, alpha=1.0):
        self.vocabulary = set()
        self.feature_counts = defaultdict(_create_default_dict_int)
        self.class_totals = defaultdict(int)
        self.class_log_probs = {}

        super().__init__(vector_store, alpha)

    def _train(self):
        """Train the Multinomial Naive Bayes classifier."""
        logger.info("Training Multinomial Naive Bayes classifier...")

        for message, label in zip(self.messages, self.labels):
            processed_text = preprocess_text(message)
            words = processed_text.split()
            word_counts = Counter(words)

            self.class_counts[label] += 1

            for word, count in word_counts.items():
                self.vocabulary.add(word)
                self.feature_counts[label][word] += count
                self.class_totals[label] += count

        # Compute class log probabilities and priors
        for label in self.class_counts:
            self.class_log_probs[label] = log(self.class_counts[label] / self.num_docs)
            self.class_priors[label] = self.class_counts[label] / self.num_docs

        logger.info("Training completed. Vocabulary size: %d", len(self.vocabulary))

    def _get_word_log_prob(self, word, label):
        """
        Calculate log probability of a word given a class with Laplace smoothing.

        Args:
            word (str): The word
            label (str): The class label

        Returns:
            float: Log probability of the word given the class
        """
        word_count = self.feature_counts[label][word]
        total_words = self.class_totals[label]
        vocab_size = len(self.vocabulary)

        # Laplace smoothing: P(word|class) = (count + alpha) / (total + alpha * vocab_size)
        prob = (word_count + self.alpha) / (total_words + self.alpha * vocab_size)
        return log(prob)

    def predict(self, query_text: str, return_probabilities=False) -> dict:
        """
        Predict the label for a given query using Naive Bayes.

        Args:
            query_text (str): The input text to classify
            return_probabilities (bool): Whether to return class probabilities

        Returns:
            dict: Contains prediction, confidence, and optionally probabilities
        """
        start_time = time.time()

        # Preprocess the query
        processed_text = preprocess_text(query_text)
        words = processed_text.split()
        word_counts = Counter(words)

        # Calculate log probabilities for each class
        class_scores = {}

        for label in self.class_counts:
            # Start with class prior probability
            score = self.class_log_probs[label]

            # Add word likelihoods
            for word, count in word_counts.items():
                if word in self.vocabulary:  # Only consider words seen during training
                    word_log_prob = self._get_word_log_prob(word, label)
                    score += count * word_log_prob

            class_scores[label] = score

        # Find the class with highest score
        predicted_label = max(class_scores, key=class_scores.get)

        # Convert log scores to probabilities for confidence
        max_score = max(class_scores.values())
        exp_scores = {label: np.exp(score - max_score) for label, score in class_scores.items()}
        total_exp = sum(exp_scores.values())
        probabilities = {label: exp_score / total_exp for label, exp_score in exp_scores.items()}

        confidence = probabilities[predicted_label]
        inference_time = time.time() - start_time

        result = {
            "prediction": predicted_label,
            "confidence": float(confidence),
            "inference_time": inference_time
        }

        if return_probabilities:
            result["probabilities"] = {label: float(prob) for label, prob in probabilities.items()}

        # Add some "neighbors" information for compatibility with other classifiers
        # For NB, we'll show the most informative words
        informative_words = self._get_informative_words(word_counts, predicted_label, top_k=5)
        result["neighbors"] = [{
            "message": f"Informative word: '{word}' (weight: {weight:.4f})",
            "label": predicted_label,
            "similarity": weight
        } for word, weight in informative_words]

        return result

    def _get_informative_words(self, word_counts, predicted_label, top_k=5):
        """
        Get the most informative words for the prediction.

        Args:
            word_counts (Counter): Word counts in the query
            predicted_label (str): The predicted class
            top_k (int): Number of top words to return

        Returns:
            list: Tuples of (word, importance_score)
        """
        word_importance = []

        for word, count in word_counts.items():
            if word in self.vocabulary:
                # Calculate the contribution of this word to the prediction
                pos_prob = self._get_word_log_prob(word, predicted_label)
                neg_label = "spam" if predicted_label == "ham" else "ham"
                neg_prob = self._get_word_log_prob(word, neg_label)

                # Importance is the difference in log probabilities
                importance = count * (pos_prob - neg_prob)
                word_importance.append((word, importance))

        # Sort by importance and return top_k
        word_importance.sort(key=lambda x: x[1], reverse=True)
        return word_importance[:top_k]

    def get_model_info(self):
        """Get model information."""
        return {
            "model_type": "Multinomial Naive Bayes",
            "num_training_docs": self.num_docs,
            "vocabulary_size": len(self.vocabulary),
            "smoothing_alpha": self.alpha,
            "class_counts": dict(self.class_counts),
            "class_priors": self.class_priors
        }


class GaussianNaiveBayes(BaseNaiveBayes):
    """
    Gaussian Naive Bayes classifier for spam/ham email classification.
    Uses TF-IDF features with Gaussian distribution assumption.
    """

    def __init__(self, vector_store=None, alpha=1.0, max_features=5000):
        self.max_features = max_features
        self.vectorizer = None
        self.feature_means = {}
        self.feature_vars = {}
        self.n_features = 0

        super().__init__(vector_store, alpha)

    def _train(self):
        """Train the Gaussian Naive Bayes classifier."""
        logger.info("Training Gaussian Naive Bayes classifier...")

        processed_messages = [preprocess_text(msg) for msg in self.messages]

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )

        # Fit and transform training data
        X = self.vectorizer.fit_transform(processed_messages).toarray()
        self.n_features = X.shape[1]

        # Calculate class statistics
        for label in set(self.labels):
            class_indices = [i for i, l in enumerate(self.labels) if l == label]
            class_features = X[class_indices]

            self.class_counts[label] = len(class_indices)
            self.class_priors[label] = len(class_indices) / self.num_docs

            # Calculate mean and variance for each feature
            self.feature_means[label] = np.mean(class_features, axis=0)
            self.feature_vars[label] = np.var(class_features, axis=0) + self.alpha

        logger.info("Training completed. Features: %d", self.n_features)

    def predict(self, query_text: str, return_probabilities=False) -> dict:
        """Predict the label using Gaussian Naive Bayes."""
        start_time = time.time()

        processed_text = preprocess_text(query_text)
        X = self.vectorizer.transform([processed_text]).toarray()[0]

        class_scores = {}

        for label in self.class_counts:
            # Start with log prior
            score = log(self.class_priors[label])

            # Add log likelihood for each feature
            means = self.feature_means[label]
            vars_ = self.feature_vars[label]

            for i in range(self.n_features):
                feature_value = X[i]
                mean = means[i]
                var = vars_[i]

                # Gaussian log probability
                log_prob = -0.5 * log(2 * pi * var) - ((feature_value - mean) ** 2) / (2 * var)
                score += log_prob

            class_scores[label] = score

        predicted_label = max(class_scores, key=class_scores.get)

        # Convert to probabilities
        max_score = max(class_scores.values())
        exp_scores = {label: np.exp(score - max_score) for label, score in class_scores.items()}
        total_exp = sum(exp_scores.values())
        probabilities = {label: exp_score / total_exp for label, exp_score in exp_scores.items()}

        confidence = probabilities[predicted_label]
        inference_time = time.time() - start_time

        result = {
            "prediction": predicted_label,
            "confidence": float(confidence),
            "inference_time": inference_time
        }

        if return_probabilities:
            result["probabilities"] = {label: float(prob) for label, prob in probabilities.items()}

        # Add informative features
        informative_features = self._get_informative_features(X, predicted_label)
        result["neighbors"] = [{
            "message": f"Feature: '{feature}' (weight: {weight:.4f})",
            "label": predicted_label,
            "similarity": weight
        } for feature, weight in informative_features[:5]]

        return result

    def _get_informative_features(self, X, predicted_label, top_k=5):
        """Get the most informative features for the prediction."""
        if self.vectorizer is None:
            return []

        feature_names = self.vectorizer.get_feature_names_out()
        feature_importance = []

        for i in range(self.n_features):
            if X[i] > 0:
                feature_value = X[i]
                feature_name = feature_names[i]

                # Calculate feature contribution difference between classes
                pos_mean = self.feature_means[predicted_label][i]
                pos_var = self.feature_vars[predicted_label][i]

                other_label = "spam" if predicted_label == "ham" else "ham"
                neg_mean = self.feature_means[other_label][i]
                neg_var = self.feature_vars[other_label][i]

                pos_log_prob = -0.5 * log(2 * pi * pos_var) - ((feature_value - pos_mean) ** 2) / (2 * pos_var)
                neg_log_prob = -0.5 * log(2 * pi * neg_var) - ((feature_value - neg_mean) ** 2) / (2 * neg_var)

                importance = feature_value * (pos_log_prob - neg_log_prob)
                feature_importance.append((feature_name, importance))

        feature_importance.sort(key=lambda x: x[1], reverse=True)
        return feature_importance[:top_k]

    def get_model_info(self):
        """Get model information."""
        return {
            "model_type": "Gaussian Naive Bayes",
            "num_training_docs": self.num_docs,
            "n_features": self.n_features,
            "max_features": self.max_features,
            "smoothing_alpha": self.alpha,
            "class_counts": dict(self.class_counts),
            "class_priors": self.class_priors
        }