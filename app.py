from flask import Flask, request, jsonify,send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import json
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.svm import SVC, SVR
import traceback
import os # Import os to get environment variables

# Import your ML functions (assuming they're in a separate file)
# If they're in the same file, you can include them directly
KERAS_AVAILABLE = False
try:
    from keras.layers import Dense
    from keras import Sequential
    from keras.utils import to_categorical
    KERAS_AVAILABLE = True
except ImportError:
    print("Keras/TensorFlow not available. Neural network models will be disabled.")
    print("To enable Keras models, ensure tensorflow is installed and configured correctly.")


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# Your ML functions (copied from your original code)
def pca(X, k):
    # Handle the case where X is empty or k is greater than the number of features
    if X.shape[0] == 0 or X.shape[1] == 0 or k == 0:
        return np.array([]).reshape(X.shape[0], 0), np.array([])
    if k > X.shape[1]:
        k = X.shape[1] # Adjust k if it's larger than the number of features

    x_standardised = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10) # Add epsilon to prevent division by zero
    cov_matrix = np.cov(x_standardised, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    ordering = np.argsort(eigenvalues)[::-1]
    selected_vectors = eigenvectors[:, ordering[:k]]
    z = x_standardised @ selected_vectors
    return z, selected_vectors

def lin_reg_scratch(X, y, epochs=50, batch_size=1, L=0.01, gd='batch'):
    y = y.reshape(-1, 1) # Ensure y is 2D
    l = [] # To store loss history
    m = np.zeros(X.shape[1]) # Coefficients
    b = 0 # Intercept

    for i in range(epochs):
        if gd == 'batch':
            y_pred = X @ m + b
            error = y - y_pred.reshape(-1, 1)
            grad_m = (-2 / len(y)) * (X.T @ error).flatten()
            grad_b = (-2 / len(y)) * np.sum(error)
            m -= L * grad_m
            b -= L * grad_b
        elif gd == 'stochastic':
            # Shuffle data for stochastic GD
            indices = np.random.permutation(len(y))
            x_shuffled = X[indices]
            y_shuffled = y[indices]
            for j in range(len(y_shuffled)):
                y_pred_single = np.dot(x_shuffled[j], m) + b
                error_single = y_shuffled[j] - y_pred_single
                m -= L * (-2) * x_shuffled[j] * error_single
                b -= L * (-2) * error_single
        elif gd == 'mini-batch':
            indices = np.random.permutation(len(y))
            x_shuffled = X[indices]
            y_shuffled = y[indices]
            for j in range(0, len(y_shuffled), batch_size):
                xb = x_shuffled[j:j+batch_size]
                yb = y_shuffled[j:j+batch_size]
                if len(yb) == 0: continue # Skip if batch is empty
                y_pred = xb @ m + b
                error = yb - y_pred.reshape(-1, 1)
                grad_m = (-2 / len(yb)) * (xb.T @ error).flatten()
                grad_b = (-2 / len(yb)) * np.sum(error)
                m -= L * grad_m
                b -= L * grad_b

        # Calculate and record loss (Mean Squared Error)
        current_y_pred = X @ m + b
        loss = mean_squared_error(y, current_y_pred)
        l.append(loss)

    final_y_pred = X @ m + b
    final_loss = mean_squared_error(y, final_y_pred)
    return [b] + m.tolist(), l, final_y_pred.flatten().tolist(), final_loss

def lin_reg_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    train_preds = model.predict(X)
    mse = mean_squared_error(y, train_preds)
    # For consistency with other models, provide a dummy loss history
    loss_history = [mse] * 20 # Simulate constant loss for simplicity
    return model.coef_.tolist(), train_preds.tolist(), mse, loss_history

def lin_reg_nn(X, y, x_test, epochs, batch_size, n1, n2, n3):
    if not KERAS_AVAILABLE:
        raise ValueError("Keras is not available. Cannot run Neural Network Regression.")

    model = Sequential([
        Dense(n1, activation='relu', input_shape=(X.shape[1],)),
        Dense(n2, activation='relu'),
        Dense(n3, activation='linear') # Linear activation for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0)
    train_preds = model.predict(X)
    test_preds = model.predict(x_test) if x_test is not None and x_test.shape[0] > 0 else None
    mse = mean_squared_error(y, train_preds)
    return train_preds.flatten().tolist(), test_preds.flatten().tolist() if test_preds is not None else None, mse, history.history['loss']

def log_reg_scratch(X, y, epochs, L):
    y = y.reshape(-1, 1)
    m = np.zeros(X.shape[1])
    b = 0
    l = []
    for i in range(epochs):
        z = X @ m + b
        p = 1 / (1 + np.exp(-z)) # Sigmoid activation
        error = p.reshape(-1,1) - y
        m -= L * (X.T @ error / len(y)).flatten()
        b -= L * np.sum(error) / len(y)
        # Calculate Binary Cross-Entropy Loss
        loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15)) # Added epsilon for log safety
        l.append(loss)

    y_pred_proba = 1 / (1 + np.exp(-(X @ m + b)))
    y_pred = (y_pred_proba >= 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    return [b] + m.tolist(), l, y_pred.flatten().tolist(), acc, cm.tolist()

def classify_with_tree(X, x_test, y, metric='entropy', max_depth=3):
    model = DecisionTreeClassifier(criterion=metric, max_depth=max_depth)
    model.fit(X, y)
    train_preds = model.predict(X)
    test_preds = model.predict(x_test) if x_test is not None and x_test.shape[0] > 0 else None
    acc = accuracy_score(y, train_preds)
    cm = confusion_matrix(y, train_preds)
    # Decision trees don't have direct 'coefficients' or 'loss_history' in the same way as linear models/NNs
    # We can return feature importances as coefficients for display.
    feature_importances = model.feature_importances_.tolist()
    # Dummy loss history for consistency
    loss_history = [1 - acc] * 20 # Example: (1 - accuracy) as a simple loss proxy
    return train_preds.tolist(), test_preds.tolist() if test_preds is not None else None, acc, cm.tolist(), feature_importances, loss_history

def classify_with_nn(X, y, x_test, epochs, batch_size, n1, n2, n3):
    if not KERAS_AVAILABLE:
        raise ValueError("Keras is not available. Cannot run Neural Network Classification.")

    model = Sequential([
        Dense(n1, activation='relu', input_shape=(X.shape[1],)),
        Dense(n2, activation='relu'),
        Dense(n3, activation='sigmoid') # Sigmoid for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0)
    train_preds_proba = model.predict(X)
    test_preds_proba = model.predict(x_test) if x_test is not None and x_test.shape[0] > 0 else None
    train_preds_binary = (train_preds_proba >= 0.5).astype(int)
    test_preds_binary = (test_preds_proba >= 0.5).astype(int) if test_preds_proba is not None else None
    acc = accuracy_score(y, train_preds_binary)
    cm = confusion_matrix(y, train_preds_binary)
    return train_preds_binary.flatten().tolist(), test_preds_binary.flatten().tolist() if test_preds_binary is not None else None, acc, cm.tolist(), history.history['loss']

def classify_with_nn_multiclass(X, y, x_test, epochs, batch_size, n1, n2, n3):
    if not KERAS_AVAILABLE:
        raise ValueError("Keras is not available. Cannot run Multiclass Neural Network Classification.")

    y_cat = to_categorical(y)
    num_classes = y_cat.shape[1] # Determine number of output neurons based on target classes

    model = Sequential([
        Dense(n1, activation='relu', input_shape=(X.shape[1],)),
        Dense(n2, activation='relu'),
        Dense(num_classes, activation='softmax') # Softmax for multiclass classification
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X, y_cat, batch_size=batch_size, epochs=epochs, verbose=0)
    train_preds_proba = model.predict(X)
    test_preds_proba = model.predict(x_test) if x_test is not None and x_test.shape[0] > 0 else None
    train_preds = np.argmax(train_preds_proba, axis=1)
    test_preds = np.argmax(test_preds_proba, axis=1) if test_preds_proba is not None else None
    acc = accuracy_score(y, train_preds)
    cm = confusion_matrix(y, train_preds)
    return train_preds.tolist(), test_preds.tolist() if test_preds is not None else None, acc, cm.tolist(), history.history['loss']

def classify_with_model(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    train_preds = model.predict(X)
    acc = accuracy_score(y, train_preds)
    cm = confusion_matrix(y, train_preds)
    # For consistency, return coefficients and a dummy loss history
    loss_history = [1 - acc] * 20 # Example: (1 - accuracy) as a simple loss proxy
    return model.coef_.tolist(), train_preds.tolist(), acc, cm.tolist(), loss_history

def classify_with_svc(X, X_test, y):
    model = SVC()
    model.fit(X, y)
    train_pred = model.predict(X)
    test_preds = model.predict(X_test) if X_test is not None and X_test.shape[0] > 0 else None
    acc = accuracy_score(y, train_pred)
    cm = confusion_matrix(y, train_pred)
    # SVC does not directly provide coefficients in the same way as linear models
    # We can return None for coefficients and a dummy loss history
    loss_history = [1 - acc] * 20
    return train_pred.tolist(), test_preds.tolist() if test_preds is not None else None, acc, cm.tolist(), None, loss_history

def predict_with_svr(X, X_test, y):
    model = SVR()
    model.fit(X, y)
    train_pred = model.predict(X)
    test_preds = model.predict(X_Test) if X_Test is not None and X_Test.shape[0] > 0 else None
    mse = mean_squared_error(y, train_pred)
    # SVR does not directly provide coefficients in the same way as linear models
    # Dummy loss history for consistency
    loss_history = [mse] * 20
    return train_pred.tolist(), test_preds.tolist() if test_preds is not None else None, mse, None, loss_history

def pipeline(X, k, y, x_test=None, scaler='StandardScaler', regression=None, classification=None,
             epochs=None, learning_rate=None, batch_size=None, n=None, n1=None, n2=None, n3=None,
             metric=None, max_depth=None):

    # Apply PCA
    x_transformed, pc_components = pca(X, k)

    # Apply scaling
    sclr = None
    if scaler == 'StandardScaler':
        sclr = StandardScaler()
    elif scaler == 'MinMaxScaler':
        sclr = MinMaxScaler()

    if sclr:
        x_transformed = sclr.fit_transform(x_transformed)
        if x_test is not None and x_test.shape[0] > 0:
            # Need to transform x_test with the scaler fitted on x_transformed, then apply pca components
            x_test_scaled = sclr.transform(x_test)
            x_test_transformed = x_test_scaled @ pc_components
        else:
            x_test_transformed = None
    else:
        # If no scaler, just apply PCA to x_test directly
        x_test_transformed = x_test @ pc_components if x_test is not None and x_test.shape[0] > 0 else None

    # Run the specified model
    if regression == 'lin_reg_scratch with batch gd':
        # lin_reg_scratch returns: [b] + m.tolist(), l, y_pred, final_loss
        coefs_and_intercept, loss_hist, train_preds, final_loss = lin_reg_scratch(x_transformed, y, epochs, L=learning_rate, gd='batch')
        return {
            "coefficients": coefs_and_intercept[1:], # Exclude intercept for now, it's handled separately
            "intercept": coefs_and_intercept[0],
            "train_predictions": train_preds,
            "training_mse": final_loss,
            "loss_history": loss_hist,
            "test_predictions": None # Scratch models don't handle test predictions
        }
    elif regression == 'lin_reg_scratch with stochastic gd':
        coefs_and_intercept, loss_hist, train_preds, final_loss = lin_reg_scratch(x_transformed, y, epochs, L=learning_rate, gd='stochastic')
        return {
            "coefficients": coefs_and_intercept[1:],
            "intercept": coefs_and_intercept[0],
            "train_predictions": train_preds,
            "training_mse": final_loss,
            "loss_history": loss_hist,
            "test_predictions": None
        }
    elif regression == 'lin_reg_scratch with mini-batch gd':
        coefs_and_intercept, loss_hist, train_preds, final_loss = lin_reg_scratch(x_transformed, y, epochs, batch_size, learning_rate, 'mini-batch')
        return {
            "coefficients": coefs_and_intercept[1:],
            "intercept": coefs_and_intercept[0],
            "train_predictions": train_preds,
            "training_mse": final_loss,
            "loss_history": loss_hist,
            "test_predictions": None
        }
    elif regression == 'lin_reg_with_model':
        # lin_reg_model returns: model.coef_, train_preds, mse, loss_history
        coefs, train_preds, mse, loss_hist = lin_reg_model(x_transformed, y)
        return {
            "coefficients": coefs,
            "train_predictions": train_preds,
            "training_mse": mse,
            "loss_history": loss_hist,
            "test_predictions": None # Sklearn model does not predict on test by default here
        }
    elif regression == 'lin_reg_nn':
        # lin_reg_nn returns: train_preds, test_preds, mse, loss_history
        train_preds, test_preds, mse, loss_hist = lin_reg_nn(x_transformed, y, x_test_transformed, epochs, batch_size, n1, n2, n3)
        return {
            "train_predictions": train_preds,
            "test_predictions": test_preds,
            "training_mse": mse,
            "loss_history": loss_hist,
            "coefficients": None # NNs don't have simple coefficients
        }
    elif classification == 'log_reg_scratch':
        # log_reg_scratch returns: [b] + m.tolist(), l, y_pred, acc, cm
        params, loss_hist, train_preds, acc, cm = log_reg_scratch(x_transformed, y, epochs, learning_rate)
        return {
            "coefficients": params[1:], # Exclude intercept
            "intercept": params[0],
            "train_predictions": train_preds,
            "training_accuracy": acc,
            "confusion_matrix": cm,
            "loss_history": loss_hist,
            "test_predictions": None
        }
    elif classification == 'log_reg_withmodel':
        # classify_with_model returns: model.coef_, train_preds, acc, cm, loss_history
        coefs, train_preds, acc, cm, loss_hist = classify_with_model(x_transformed, y)
        return {
            "coefficients": coefs,
            "train_predictions": train_preds,
            "training_accuracy": acc,
            "confusion_matrix": cm,
            "loss_history": loss_hist,
            "test_predictions": None # Sklearn model does not predict on test by default here
        }
    elif classification == 'log_regwithnn':
        # classify_with_nn returns: train_preds_binary, test_preds_binary, acc, cm, loss_history
        train_preds, test_preds, acc, cm, loss_hist = classify_with_nn(x_transformed, y, x_test_transformed, epochs, batch_size, n1, n2, n3)
        return {
            "train_predictions": train_preds,
            "test_predictions": test_preds,
            "training_accuracy": acc,
            "confusion_matrix": cm,
            "loss_history": loss_hist,
            "coefficients": None # NNs don't have simple coefficients
        }
    elif classification == 'classify with tree':
        # classify_with_tree returns: train_preds, test_preds, acc, cm, feature_importances, loss_history
        train_preds, test_preds, acc, cm, feature_importances, loss_hist = classify_with_tree(x_transformed, x_test_transformed, y, metric, max_depth)
        return {
            "train_predictions": train_preds,
            "test_predictions": test_preds,
            "training_accuracy": acc,
            "confusion_matrix": cm,
            "coefficients": feature_importances, # Use feature importances as coefficients for trees
            "loss_history": loss_hist
        }
    elif classification == 'classify with SVC':
        # classify_with_svc returns: train_pred, test_preds, acc, cm, None, loss_history
        train_preds, test_preds, acc, cm, _, loss_hist = classify_with_svc(x_transformed, x_test_transformed, y)
        return {
            "train_predictions": train_preds,
            "test_predictions": test_preds,
            "training_accuracy": acc,
            "confusion_matrix": cm,
            "coefficients": None, # SVC does not have simple coefficients
            "loss_history": loss_hist
        }
    elif classification == 'classify with nn multiclass':
        # classify_with_nn_multiclass returns: train_preds.tolist(), test_preds.tolist(), acc, cm, history.history['loss']
        train_preds, test_preds, acc, cm, loss_hist = classify_with_nn_multiclass(x_transformed, y, x_test_transformed, epochs, batch_size, n1, n2, n3)
        return {
            "train_predictions": train_preds,
            "test_predictions": test_preds,
            "training_accuracy": acc,
            "confusion_matrix": cm,
            "loss_history": loss_hist,
            "coefficients": None # NNs don't have simple coefficients
        }
    elif regression == 'regression with svr':
        # predict_with_svr returns: train_pred, test_preds, mse, None, loss_history
        train_preds, test_preds, mse, _, loss_hist = predict_with_svr(x_transformed, x_test_transformed, y)
        return {
            "train_predictions": train_preds,
            "test_predictions": test_preds,
            "training_mse": mse,
            "coefficients": None, # SVR does not have simple coefficients
            "loss_history": loss_hist
        }
    else:
        raise ValueError("Invalid regression or classification model specified.")


def parse_csv_data(csv_data):
    """Parse CSV data from frontend format to numpy arrays"""
    headers = csv_data['headers']
    rows = csv_data['rows']

    # Convert to DataFrame first for easier handling
    df = pd.DataFrame(rows, columns=headers)

    # Convert to numeric where possible, coercing non-numeric to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN values introduced by coercion (or handle as appropriate for your data)
    # For ML, often NaN values need imputation or rows dropped. Here, we'll drop for simplicity
    df = df.dropna()

    return df.values, headers

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "keras_available": KERAS_AVAILABLE})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Parse input data
        x_train_data = data['x_train']
        y_train_data = data['y_train']
        x_test_data = data.get('x_test', None)
        config = data['config']

        # Convert CSV data to numpy arrays
        X_train, x_headers = parse_csv_data(x_train_data)
        y_train, y_headers = parse_csv_data(y_train_data)

        # Flatten y_train if it's 2D with single column
        if y_train.shape[1] == 1:
            y_train = y_train.flatten()

        X_test = None
        if x_test_data and len(x_test_data['rows']) > 0: # Ensure test data is actually provided
            X_test, _ = parse_csv_data(x_test_data)

        # Extract configuration
        k = config.get('k', 2)
        scaler = config.get('scaler', 'StandardScaler')
        epochs = config.get('epochs', 100)
        learning_rate = config.get('learning_rate', 0.01)
        batch_size = config.get('batch_size', 32)
        n1 = config.get('n1', 64)
        n2 = config.get('n2', 32)
        n3 = config.get('n3', 1)
        metric = config.get('metric', 'entropy')
        max_depth = config.get('max_depth', 3)
        regression = config.get('regression', None)
        classification = config.get('classification', None)

        # Ensure k doesn't exceed number of features
        k = min(k, X_train.shape[1])

        # Ensure n3 (output layer neurons) is appropriate for classification tasks
        if classification in ['log_regwithnn', 'classify with nn multiclass']:
            # For classification, n3 should be 1 for binary, or number of unique classes for multiclass
            if classification == 'log_regwithnn':
                 n3 = 1 # Binary classification
            else: # classify with nn multiclass
                n3 = len(np.unique(y_train)) # Number of unique classes

        # Run the pipeline
        pipeline_results = pipeline(
            X=X_train,
            k=k,
            y=y_train,
            x_test=X_test,
            scaler=scaler,
            regression=regression,
            classification=classification,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size, # Use batch_size for NNs as well
            n1=n1,
            n2=n2,
            n3=n3,
            metric=metric,
            max_depth=max_depth
        )

        response = {
            "status": "success",
            "model_type": regression if regression else classification,
            "data_info": {
                "train_shape": X_train.shape,
                "test_shape": X_test.shape if X_test is not None else None,
                "features": x_headers,
                "target": y_headers[0] if len(y_headers) > 0 else "target"
            }
        }

        # Merge pipeline_results into the main response
        response.update(pipeline_results)

        return jsonify(response)

    except ValueError as ve:
        error_msg = str(ve)
        print(f"Value Error: {error_msg}")
        return jsonify({
            "status": "error",
            "error": error_msg,
            "traceback": traceback.format_exc()
        }), 400 # Bad request for value errors
    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error: {error_msg}")
        print(f"Traceback: {traceback_str}")

        return jsonify({
            "status": "error",
            "error": "An internal server error occurred: " + error_msg,
            "traceback": traceback_str
        }), 500

@app.route('/models', methods=['GET'])
def get_available_models():
    """Return available models based on dependencies"""
    models = {
        "regression": [
            "lin_reg_with_model",
            "lin_reg_scratch with batch gd",
            "lin_reg_scratch with stochastic gd",
            "lin_reg_scratch with mini-batch gd",
            "regression with svr"
        ],
        "classification": [
            "log_reg_withmodel",
            "log_reg_scratch",
            "classify with tree",
            "classify with SVC"
        ]
    }

    if KERAS_AVAILABLE:
        models["regression"].append("lin_reg_nn")
        models["classification"].extend([
            "log_regwithnn",
            "classify with nn multiclass"
        ])

    return jsonify({
        "status": "success",
        "models": models,
        "keras_available": KERAS_AVAILABLE
    })

if __name__ == '__main__':
    port_str = os.environ.get("PORT")
    if port_str and port_str.isdigit(): # Check if it's not empty and is a digit string
        port = int(port_str)
    else:
        port = 5000 # Default port for local development
   
    debug_mode = os.environ.get("FLASK_DEBUG", "1") == "1" # Default to debug true for local
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
