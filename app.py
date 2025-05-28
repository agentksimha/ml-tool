from flask import Flask, request, jsonify
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

# Import your ML functions (assuming they're in a separate file)
# If they're in the same file, you can include them directly
try:
    from keras.layers import Dense
    from keras import Sequential
    from keras.utils import to_categorical
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Keras not available. Neural network models will be disabled.")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/")
# Your ML functions (copied from your original code)
def pca(X, k):
    x_standardised = (X - np.mean(X, axis=0)) / (np.std(X, axis=0))
    cov_matrix = np.cov(x_standardised, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    ordering = np.argsort(eigenvalues)[::-1]
    selected_vectors = eigenvectors[:, ordering[:k]]
    z = x_standardised @ selected_vectors
    return z, selected_vectors

def lin_reg_scratch(X, y, epochs=50, batch_size=1, L=0.01, gd='batch'):
    y = y.reshape(-1, 1)
    l = []
    m = np.zeros(X.shape[1])
    b = 0
    for i in range(epochs):
        if gd == 'batch':
            y_pred = X @ m + b
            error = y - y_pred.reshape(-1, 1)
            m -= L * (-2 / len(y)) * (X.T @ error).flatten()
            b -= L * (-2 / len(y)) * np.sum(error)
        elif gd == 'stochastic':
            for j in range(len(y)):
                y_pred = np.dot(X[j], m) + b
                error = y[j] - y_pred
                m -= L * (-2) * X[j] * error
                b -= L * (-2) * error
        elif gd == 'mini-batch':
            indices = np.random.permutation(len(y))
            x_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0, len(y), batch_size):
                xb = x_shuffled[i:i+batch_size]
                yb = y_shuffled[i:i+batch_size]
                y_pred = xb @ m + b
                error = yb - y_pred.reshape(-1, 1)
                m -= L * (-2 / len(yb)) * (xb.T @ error).flatten()
                b -= L * (-2 / len(yb)) * np.sum(error)
        loss = -np.mean(y * np.log(p.reshape(-1,1) + 1e-15) + (1 - y) * np.log(1 - p.reshape(-1,1) + 1e-15))
        l.append(loss)
    y_pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    return [b] + m.tolist(), l, acc, cm

def lin_reg_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    train_preds = model.predict(X)
    acc = mean_squared_error(y, train_preds)
    return model.coef_, train_preds, acc

def lin_reg_nn(X, y, x_test, epochs, n, n1, n2, n3):
    model = Sequential([
        Dense(n1, activation='relu', input_shape=(X.shape[1],)),
        Dense(n2, activation='relu'),
        Dense(n3, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, n, epochs, verbose=0)
    train_preds = model.predict(X)
    test_preds = model.predict(x_test)
    acc = mean_squared_error(y, train_preds)
    return train_preds, test_preds, acc

def log_reg_scratch(X, y, epochs, L):
    y = y.reshape(-1, 1)
    m = np.zeros(X.shape[1])
    b = 0
    l = []
    for i in range(epochs):
        z = X @ m + b
        p = 1 / (1 + np.exp(-z))
        error = p.reshape(-1,1) - y
        m -= L * (X.T @ error / len(y)).flatten()
        b -= L * np.sum(error) / len(y)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        l.append(loss)
    y_pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    return [b] + m.tolist(), l, acc, cm

def classify_with_tree(X, x_test, y, metric='entropy', max_depth=3):
    model = DecisionTreeClassifier(criterion=metric, max_depth=max_depth)
    model.fit(X, y)
    train_preds = model.predict(X)
    test_preds = model.predict(x_test) if x_test is not None else None
    acc = accuracy_score(y, train_preds)
    cm = confusion_matrix(y, train_preds)
    return train_preds, test_preds, acc, cm

def classify_with_nn(X, y, x_test, epochs, n, n1, n2, n3):
    if not KERAS_AVAILABLE:
        raise ValueError("Keras is not available. Please install tensorflow/keras.")
    
    model = Sequential([
        Dense(n1, activation='relu', input_shape=(X.shape[1],)),
        Dense(n2, activation='relu'),
        Dense(n3, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    history = model.fit(X, y, batch_size=n, epochs=epochs, verbose=0)
    train_preds = model.predict(X)
    test_preds = model.predict(x_test) if x_test is not None else None
    train_preds_binary = (train_preds >= 0.5).astype(int)
    acc = accuracy_score(y, train_preds_binary)
    cm = confusion_matrix(y, train_preds_binary)
    return train_preds, test_preds, acc, cm, history.history['loss']

def classify_with_nn_multiclass(X, y, x_test, epochs, n, n1, n2, n3):
    if not KERAS_AVAILABLE:
        raise ValueError("Keras is not available. Please install tensorflow/keras.")
    
    y_cat = to_categorical(y)
    model = Sequential([
        Dense(n1, activation='relu', input_shape=(X.shape[1],)),
        Dense(n2, activation='relu'),
        Dense(n3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X, y_cat, batch_size=n, epochs=epochs, verbose=0)
    train_preds_proba = model.predict(X)
    test_preds_proba = model.predict(x_test) if x_test is not None else None
    train_preds = np.argmax(train_preds_proba, axis=1)
    test_preds = np.argmax(test_preds_proba, axis=1) if test_preds_proba is not None else None
    acc = accuracy_score(y, train_preds)
    cm = confusion_matrix(y, train_preds)
    return train_preds.tolist(), test_preds.tolist() if test_preds is not None else None, acc, cm, history.history['loss']

def classify_with_model(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    train_preds = model.predict(X)
    acc = accuracy_score(y, train_preds)
    cm = confusion_matrix(y, train_preds)
    return model.coef_, train_preds, acc, cm

def classify_with_svc(X, X_test, y):
    model = SVC()
    model.fit(X, y)
    train_pred = model.predict(X)
    test_preds = model.predict(X_test) if X_test is not None else None
    acc = accuracy_score(y, train_pred)
    cm = confusion_matrix(y, train_pred)
    return train_pred, test_preds, acc, cm

def predict_with_svr(X, X_test, y):
    model = SVR()
    model.fit(X, y)
    train_pred = model.predict(X)
    test_preds = model.predict(X_test) if X_test is not None else None
    l = mean_squared_error(y, train_pred)
    return train_pred, test_preds, l

def pipeline(X, k, y, x_test=None, scaler='StandardScaler', regression=None, classification=None,
             epochs=None, learning_rate=None, batch_size=None, n=None, n1=None, n2=None, n3=None,
             metric=None, max_depth=None):
    
    # Apply PCA
    x_transformed, pc = pca(X, k)

    # Apply scaling
    if scaler == 'StandardScaler':
        sclr = StandardScaler()
    elif scaler == 'MinMaxScaler':
        sclr = MinMaxScaler()
    else:
        sclr = None

    if sclr:
        x_transformed = sclr.fit_transform(x_transformed)
        if x_test is not None:
            x_test1 = sclr.transform(x_test)
            x_test_transformed = x_test1 @ pc
        else:
            x_test_transformed = None
    else:
        x_test_transformed = x_test @ pc if x_test is not None else None

    # Run the specified model
    if regression == 'lin_reg_scratch with batch gd':
        return lin_reg_scratch(x_transformed, y, epochs, learning_rate=learning_rate, gd='batch')
    elif regression == 'lin_reg_scratch with stochastic gd':
        return lin_reg_scratch(x_transformed, y, epochs, learning_rate=learning_rate, gd='stochastic')
    elif regression == 'lin_reg_scratch with mini-batch gd':
        return lin_reg_scratch(x_transformed, y, epochs, batch_size, learning_rate, 'mini-batch')
    elif regression == 'lin_reg_with_model':
        return lin_reg_model(x_transformed, y)
    elif regression == 'lin_reg_nn':
        return lin_reg_nn(x_transformed, y, x_test_transformed, epochs, n, n1, n2, n3)
    elif classification == 'log_reg_scratch':
        return log_reg_scratch(x_transformed, y, epochs, learning_rate)
    elif classification == 'log_reg_withmodel':
        return classify_with_model(x_transformed, y)
    elif classification == 'log_regwithnn':
        return classify_with_nn(x_transformed, y, x_test_transformed, epochs, n, n1, n2, n3)
    elif classification == 'classify with tree':
        return classify_with_tree(x_transformed, x_test_transformed, y, metric, max_depth)
    elif classification == 'classify with SVC':
        return classify_with_svc(x_transformed, x_test_transformed, y)
    elif classification == 'classify with nn multiclass':
        return classify_with_nn_multiclass(x_transformed, y, x_test_transformed, epochs, n, n1, n2, n3)
    elif regression == 'regression with svr':
        return predict_with_svr(x_transformed, x_test_transformed, y)

def parse_csv_data(csv_data):
    """Parse CSV data from frontend format to numpy arrays"""
    headers = csv_data['headers']
    rows = csv_data['rows']
    
    # Convert to DataFrame first for easier handling
    df = pd.DataFrame(rows, columns=headers)
    
    # Convert to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
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
        if x_test_data:
            X_test, _ = parse_csv_data(x_test_data)
        
        # Extract configuration
        k = config.get('k', 2)
        scaler = config.get('scaler', 'StandardScaler')
        epochs = config.get('epochs', 100)
        learning_rate = config.get('learning_rate', 0.01)
        batch_size = config.get('batch_size', 32)
        n = config.get('n', 32)
        n1 = config.get('n1', 64)
        n2 = config.get('n2', 32)
        n3 = config.get('n3', 1)
        metric = config.get('metric', 'entropy')
        max_depth = config.get('max_depth', 3)
        regression = config.get('regression', None)
        classification = config.get('classification', None)
        
        # Ensure k doesn't exceed number of features
        k = min(k, X_train.shape[1])
        
        # Run the pipeline
        result = pipeline(
            X=X_train,
            k=k,
            y=y_train,
            x_test=X_test,
            scaler=scaler,
            regression=regression,
            classification=classification,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n=n,
            n1=n1,
            n2=n2,
            n3=n3,
            metric=metric,
            max_depth=max_depth
        )
        
        # Format response based on model type
        response = {
            "status": "success",
            "model_type": regression or classification,
            "data_info": {
                "train_shape": X_train.shape,
                "test_shape": X_test.shape if X_test is not None else None,
                "features": x_headers,
                "target": y_headers[0] if len(y_headers) > 0 else "target"
            }
        }
        
        # Handle different return formats
        if regression:
            if regression in ['lin_reg_scratch with batch gd', 'lin_reg_scratch with stochastic gd', 'lin_reg_scratch with mini-batch gd']:
                m, b, y_pred, loss_history, final_loss = result
                response.update({
                    "coefficients": m.tolist() if hasattr(m, 'tolist') else m,
                    "intercept": float(b),
                    "train_predictions": y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
                    "loss_history": loss_history,
                    "final_loss": float(final_loss),
                    "training_mse": float(final_loss)
                })
            elif regression == 'lin_reg_with_model':
                coef, train_preds, mse = result
                response.update({
                    "coefficients": coef.tolist(),
                    "train_predictions": train_preds.tolist(),
                    "training_mse": float(mse)
                })
            elif regression == 'lin_reg_nn':
                train_preds, test_preds, mse, loss_history = result
                response.update({
                    "train_predictions": train_preds.flatten().tolist(),
                    "test_predictions": test_preds.flatten().tolist() if test_preds is not None else None,
                    "training_mse": float(mse),
                    "loss_history": loss_history
                })
            elif regression == 'regression with svr':
                train_preds, test_preds, mse = result
                response.update({
                    "train_predictions": train_preds.tolist(),
                    "test_predictions": test_preds.tolist() if test_preds is not None else None,
                    "training_mse": float(mse)
                })
        
        elif classification:
            if classification == 'log_reg_scratch':
                params, loss_history, acc, cm = result
                response.update({
                    "parameters": params,
                    "loss_history": loss_history,
                    "training_accuracy": float(acc),
                    "confusion_matrix": cm.tolist()
                })
            elif classification in ['log_reg_withmodel', 'classify with tree', 'classify with SVC']:
                if len(result) == 4:
                    if classification == 'log_reg_withmodel':
                        coef, train_preds, acc, cm = result
                        response.update({
                            "coefficients": coef.tolist(),
                            "train_predictions": train_preds.tolist(),
                            "training_accuracy": float(acc),
                            "confusion_matrix": cm.tolist()
                        })
                    else:
                        train_preds, test_preds, acc, cm = result
                        response.update({
                            "train_predictions": train_preds.tolist(),
                            "test_predictions": test_preds.tolist() if test_preds is not None else None,
                            "training_accuracy": float(acc),
                            "confusion_matrix": cm.tolist()
                        })
            elif classification in ['log_regwithnn', 'classify with nn multiclass']:
                if len(result) == 5:
                    train_preds, test_preds, acc, cm, loss_history = result
                    response.update({
                        "train_predictions": train_preds if isinstance(train_preds, list) else train_preds.tolist(),
                        "test_predictions": test_preds,
                        "training_accuracy": float(acc),
                        "confusion_matrix": cm.tolist(),
                        "loss_history": loss_history
                    })
                else:
                    train_preds, test_preds, acc, cm = result
                    response.update({
                        "train_predictions": train_preds.tolist(),
                        "test_predictions": test_preds.tolist() if test_preds is not None else None,
                        "training_accuracy": float(acc),
                        "confusion_matrix": cm.tolist()
                    })
        
        return jsonify(response)
    
    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error: {error_msg}")
        print(f"Traceback: {traceback_str}")
        
        return jsonify({
            "status": "error",
            "error": error_msg,
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
    app.run(debug=True, host='0.0.0.0', port=5000) 
