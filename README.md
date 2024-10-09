# SecureBank Fraud Detection

## Overview

The SecureBank Fraud Detection System is a comprehensive solution designed to detect and prevent fraudulent transactions in real-time. This system combines a backend built with Flask and a user-friendly frontend created with React to provide an intuitive interface for fraud detection and model management.

![SecureBank Dashboard Overview](images/dashboard_overview.png)
*Figure 1: An overview of the SecureBank Dashboard showing all available tabs.*

## Features

1. **Real-time Fraud Detection**: Predict whether a transaction is fraudulent or legitimate based on various parameters.
2. **Dataset Generation**: Create custom datasets for training and testing fraud detection models.
3. **Model Training**: Train different types of machine learning models on generated datasets.
4. **Model Selection**: Choose from various pre-trained models for fraud detection.
5. **Performance Auditing**: Evaluate the performance of selected models on different datasets.
6. **Transaction History**: View a log of past transactions and their fraud predictions.

## System Architecture

The SecureBank Fraud Detection System is built using a microservices architecture, with a Flask backend handling the core logic and a React frontend providing the user interface.

![System Architecture](images/system_architecture.png)
*Figure 2: A diagram showing the system architecture, including the Flask backend, React frontend, and Docker containers.*

### Backend (Flask)

The backend is responsible for:
- Processing transactions and making fraud predictions
- Generating synthetic datasets for training and testing
- Training and managing machine learning models
- Auditing model performance
- Providing API endpoints for the frontend

### Frontend (React)

The frontend provides a user-friendly interface for:
- Submitting transactions for fraud detection
- Generating new datasets
- Training and selecting models
- Viewing transaction history
- Auditing model performance

## User Interface

The user interface is divided into five main sections, each accessible via a tab in the dashboard:

### 1. Predict

![Predict Tab](images/predict_tab.png)
*Figure 3: The Predict tab, showing the transaction input form and prediction result.*

This tab allows users to input transaction details and receive a fraud prediction in real-time.

### 2. Dataset

![Dataset Tab](images/dataset_tab.png)
*Figure 4: The Dataset tab, displaying options for generating new datasets.*

Users can generate new datasets by specifying parameters such as the number of transactions, fraud ratio, etc.

### 3. Model

![Model Tab](images/model_tab.png)
*Figure 5: The Model tab, showing options for training and selecting models.*

This tab provides interfaces for training new models on generated datasets and selecting models for use in fraud detection.

### 4. History

![History Tab](images/history_tab.png)
*Figure 6: The History tab, displaying a log of past transactions and their fraud predictions.*

Users can view a history of transactions processed by the system, along with their fraud predictions.

### 5. Audit

![Audit Tab](images/audit_tab.png)
*Figure 7: The Audit tab, showing performance metrics for the selected model.*

This tab allows users to audit the performance of selected models, displaying metrics such as false positive and false negative rates.

## Getting Started

To run the SecureBank Fraud Detection System:

1. Clone the repository:
   ```
   git clone https://github.com/RamseyTh/securebank.git
   ```

2. Navigate to the project directory:
   ```
   cd securebank
   ```

3. Build and run the Docker containers:
   ```
   docker-compose up --build
   ```

4. Access the application at `http://localhost:3000` in your web browser.

## Development

For development purposes, you can run the backend and frontend separately:

### Backend

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```
   python app.py
   ```

### Frontend

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install the required Node.js packages:
   ```
   npm install
   ```

3. Start the React development server:
   ```
   npm start
   ```

