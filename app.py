import pickle
import numpy as np
from flask import Flask
from flask import request
from flask import jsonify

class Perceptron:
    
    def __init__(self, eta=0.095, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.random.uniform(-1.0, 1.0, 1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

ppn = Perceptron()
import pickle
with open('model.pkl', "wb") as picklefile:
    pickle.dump(ppn, picklefile)
    
app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        sepal_length = float(request.args.get('sl'))
        petal_length = float(request.args.get('pl'))
    elif request.method == 'POST':
        data = request.get_json(force=True)
        sepal_length = float(data.get('sl'))
        petal_length = float(data.get('pl'))
    else:
        return jsonify({'error': 'Unsupported HTTP method'}), 405

    features = [sepal_length, petal_length]

    with open('model.pkl', "rb") as picklefile:
        model = pickle.load(picklefile)
        
    predicted_class = int(model.predict([features])[0])
    
    return jsonify(features=features, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
