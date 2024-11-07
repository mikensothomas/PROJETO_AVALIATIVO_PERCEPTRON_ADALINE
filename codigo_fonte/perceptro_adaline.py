import numpy as np
import pandas as pd

learning_rate = 0.01
epochs = 1000

def step_function(x):
    return 1 if x >= 0 else -1

def linear_function(x):
    return x

def train_perceptron(X, y):
    weights = np.random.rand(X.shape[1] + 1)
    initial_weights = weights.copy()
    for epoch in range(epochs):
        for i, x in enumerate(X):
            x_with_bias = np.insert(x, 0, 1)
            net_input = np.dot(weights, x_with_bias)
            prediction = step_function(net_input)
            error = y[i] - prediction
            weights += learning_rate * error * x_with_bias
        if np.all(np.vectorize(step_function)(np.dot(np.insert(X, 0, 1, axis=1), weights)) == y):
            break
    return initial_weights, weights, epoch + 1  

def train_adaline(X, y):
    weights = np.random.rand(X.shape[1] + 1)  
    initial_weights = weights.copy() 
    for epoch in range(epochs):
        errors = []
        for i, x in enumerate(X):
            x_with_bias = np.insert(x, 0, 1) 
            net_input = np.dot(weights, x_with_bias)
            prediction = linear_function(net_input)
            error = y[i] - prediction
            weights += learning_rate * error * x_with_bias 
            errors.append(error ** 2)
        if np.mean(errors) < 0.01:
            break
    return initial_weights, weights, epoch + 1

def predict(X, weights, model_type="perceptron"):
    predictions = []
    for x in X:
        x_with_bias = np.insert(x, 0, 1)
        net_input = np.dot(weights, x_with_bias)
        if model_type == "perceptron":
            predictions.append(step_function(net_input))
        elif model_type == "adaline":
            predictions.append(linear_function(net_input))
    return np.array(predictions)

X_train = np.array([
    [0.5, 0.2, 0.1],
    [0.9, 0.4, 0.7],
    [0.4, 0.6, 0.5],
    [0.6, 0.7, 0.2]
])

y_train = np.array([-1, 1, -1, 1])

X_test = np.array([
    [0.5, 0.3, 0.8],
    [0.2, 0.6, 0.1],
    [0.9, 0.7, 0.5]
])

results_perceptron = []
results_adaline = []

for i in range(5):
    initial_weights_perceptron, final_weights_perceptron, epochs_perceptron = train_perceptron(X_train, y_train)
    results_perceptron.append((initial_weights_perceptron, final_weights_perceptron, epochs_perceptron))
    
    initial_weights_adaline, final_weights_adaline, epochs_adaline = train_adaline(X_train, y_train)
    results_adaline.append((initial_weights_adaline, final_weights_adaline, epochs_adaline))

table_3_2_perceptron = pd.DataFrame({
    "Treinamento": [f"T{i+1}" for i in range(5)],
    "Pesos Iniciais (w0, w1, w2, w3)": [result[0] for result in results_perceptron],
    "Pesos Finais (w0, w1, w2, w3)": [result[1] for result in results_perceptron],
    "Número de Épocas": [result[2] for result in results_perceptron]
})

table_3_2_adaline = pd.DataFrame({
    "Treinamento": [f"T{i+1}" for i in range(5)],
    "Pesos Iniciais (w0, w1, w2, w3)": [result[0] for result in results_adaline],
    "Pesos Finais (w0, w1, w2, w3)": [result[1] for result in results_adaline],
    "Número de Épocas": [result[2] for result in results_adaline]
})

predictions_perceptron = [predict(X_test, result[1], model_type="perceptron") for result in results_perceptron]
predictions_adaline = [predict(X_test, result[1], model_type="adaline") for result in results_adaline]

table_3_3_perceptron = pd.DataFrame({
    "Amostra": [f"Amostra {i+1}" for i in range(len(X_test))],
    **{f"y (T{i+1})": predictions_perceptron[i] for i in range(5)}
})

table_3_3_adaline = pd.DataFrame({
    "Amostra": [f"Amostra {i+1}" for i in range(len(X_test))],
    **{f"y (T{i+1})": predictions_adaline[i] for i in range(5)}
})

print("Tabela 3.2 - Resultados Perceptron")
print(table_3_2_perceptron)

print("\nTabela 3.2 - Resultados Adaline")
print(table_3_2_adaline)

print("\nTabela 3.3 - Classificação das Amostras (Perceptron)")
print(table_3_3_perceptron)

print("\nTabela 3.3 - Classificação das Amostras (Adaline)")
print(table_3_3_adaline)

table_3_2_perceptron.to_csv("Tabela_3_2_Perceptron.csv", index=False)
table_3_2_adaline.to_csv("Tabela_3_2_Adaline.csv", index=False)
table_3_3_perceptron.to_csv("Tabela_3_3_Perceptron.csv", index=False)
table_3_3_adaline.to_csv("Tabela_3_3_Adaline.csv", index=False)