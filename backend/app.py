from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from modules.pipeline import Pipeline
from modules.data_generator import DataGenerator
from modules.model_trainer import ModelTrainer
from modules.performance_auditor import PerformanceAuditor
import time

app = Flask(__name__)
CORS(app)  

pipeline = Pipeline()
data_generator = DataGenerator()
model_trainer = ModelTrainer()
performance_auditor = PerformanceAuditor()

@app.route('/predict/', methods=['POST'])
def predict():
    data = request.json
    required_keys = [
        'trans_date_trans_time', 'cc_num', 'unix_time', 'merchant',
        'category', 'amt', 'merch_lat', 'merch_long'
    ]
    
    if not all(key in data for key in required_keys):
        return jsonify({"error": "Missing required fields"}), 400
    try:
        prediction = pipeline.predict(data)
        return jsonify({"prediction": "fraud" if prediction else "legitimate"})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/generate_dataset/', methods=['POST'])
def generate_dataset():
    data = request.json
    version = data.get('version')
    num_customers = data.get('num_customers', 1000)
    num_transactions = data.get('num_transactions', 10000)
    fraud_ratio = data.get('fraud_ratio', 0.01)
    
    if not version:
        return jsonify({"error": "Missing version for dataset"}), 400
    try:
        data_generator.generate_and_save_data(version=version, num_customers=num_customers, 
                                              num_transactions=num_transactions, fraud_ratio=fraud_ratio)
        return jsonify({"message": f"Dataset version {version} generated successfully"})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/train_model/', methods=['POST'])
def train_model():
    data = request.json
    model_name = data.get('model_name')
    dataset_version = data.get('dataset_version')

    if not model_name or not dataset_version:
        return jsonify({"error": "Missing model name or dataset version"}), 400
    try:
        model_trainer.train(model_name, dataset_version)
        return jsonify({"message": f"Model {model_name} trained successfully on dataset {dataset_version}"})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/select_model/', methods=['POST'])
def select_model():
    data = request.json
    model_name = data.get('model_name')
    if not model_name:
        return jsonify({"error": "Missing model name"}), 400
    try:
        pipeline.select_model(model_name)
        return jsonify({"message": f"Model {model_name} selected successfully"})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/history', methods=['GET'])
def get_history():
    history = pipeline.get_history()
    return jsonify([{"transaction": str(transaction), "prediction": "fraud" if prediction else "legitimate"} 
                    for transaction, prediction in history])

@app.route('/audit_performance/', methods=['POST'])
def audit_performance():
    data = request.json
    dataset_version = data.get('dataset_version')

    if not dataset_version:
        return jsonify({"error": "Missing dataset version"}), 400

    try:
        performance_metrics = performance_auditor.audit(pipeline, dataset_version)
        return jsonify(performance_metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/datasets', methods=['GET'])
def get_datasets():
    datasets = ['v1.0', 'v1.1', 'v2.0']
    return jsonify(datasets)

@app.route('/models', methods=['GET'])
def get_models():
    models = ['logistic_regression', 'rvm', 'random_forest']
    return jsonify(models)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)