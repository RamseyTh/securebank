import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AlertCircle, CheckCircle } from 'lucide-react';

const API_URL = 'http://localhost:5001';

const SecureBankDashboard = () => {
  const [activeTab, setActiveTab] = useState('predict');
  const [prediction, setPrediction] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState(['logistic_regression', 'rvm', 'random_forest']);
  const [selectedModel, setSelectedModel] = useState('');
  const [history, setHistory] = useState([]);
  const [auditResults, setAuditResults] = useState(null);

  const [transactionData, setTransactionData] = useState({
    trans_date_trans_time: '',
    cc_num: '',
    unix_time: '',
    merchant: '',
    category: '',
    amt: '',
    merch_lat: '',
    merch_long: ''
  });

  const [datasetParams, setDatasetParams] = useState({
    version: '',
    num_customers: 100,
    num_transactions: 1000,
    fraud_ratio: 0.01
  });

  useEffect(() => {
    fetchHistory();
    fetchDatasets();
  }, []);

  const fetchHistory = async () => {
    try {
      const response = await fetch(`${API_URL}/history`);
      const data = await response.json();
      setHistory(data);
    } catch (error) {
      console.error('Error fetching history:', error);
    }
  };

  const fetchDatasets = async () => {
    try {
      const response = await fetch(`${API_URL}/datasets`);
      const data = await response.json();
      setDatasets(data);
    } catch (error) {
      console.error('Error fetching datasets:', error);
    }
  };

  const handleInputChange = (e) => {
    setTransactionData({ ...transactionData, [e.target.name]: e.target.value });
  };

  const handleDatasetParamChange = (e) => {
    setDatasetParams({ ...datasetParams, [e.target.name]: e.target.value });
  };

  const predictTransaction = async () => {
    try {
      const response = await fetch(`${API_URL}/predict/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(transactionData)
      });
      const data = await response.json();
      setPrediction(data.prediction);
      fetchHistory();
    } catch (error) {
      console.error('Error predicting transaction:', error);
    }
  };

  const generateDataset = async () => {
    try {
      const response = await fetch(`${API_URL}/generate_dataset/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(datasetParams)
      });
      const data = await response.json();
      console.log(data);
      fetchDatasets();
    } catch (error) {
      console.error('Error generating dataset:', error);
    }
  };

  const trainModel = async () => {
    try {
      const response = await fetch(`${API_URL}/train_model/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: selectedModel, dataset_version: datasetParams.version })
      });
      const data = await response.json();
      console.log(data);
    } catch (error) {
      console.error('Error training model:', error);
    }
  };

  const selectModel = async () => {
    try {
      const response = await fetch(`${API_URL}/select_model/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: selectedModel })
      });
      const data = await response.json();
      console.log(data);
    } catch (error) {
      console.error('Error selecting model:', error);
    }
  };

  const auditPerformance = async () => {
    try {
      const response = await fetch(`${API_URL}/audit_performance/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_version: datasetParams.version })
      });
      const data = await response.json();
      setAuditResults(data);
    } catch (error) {
      console.error('Error auditing performance:', error);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="predict">Predict</TabsTrigger>
          <TabsTrigger value="dataset">Dataset</TabsTrigger>
          <TabsTrigger value="model">Model</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
          <TabsTrigger value="audit">Audit</TabsTrigger>
        </TabsList>
        <TabsContent value="predict">
          <Card>
            <CardHeader>Predict Transaction</CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                {Object.keys(transactionData).map(key => (
                  <Input
                    key={key}
                    name={key}
                    placeholder={key}
                    value={transactionData[key]}
                    onChange={handleInputChange}
                  />
                ))}
              </div>
              <Button className="mt-4" onClick={predictTransaction}>Predict</Button>
              {prediction && (
                <div className="mt-4 flex items-center">
                  {prediction === 'fraud' ? (
                    <AlertCircle className="text-red-500 mr-2" />
                  ) : (
                    <CheckCircle className="text-green-500 mr-2" />
                  )}
                  <span>Prediction: {prediction}</span>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="dataset">
          <Card>
            <CardHeader>Generate Dataset</CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                {Object.keys(datasetParams).map(key => (
                  <Input
                    key={key}
                    name={key}
                    placeholder={key}
                    value={datasetParams[key]}
                    onChange={handleDatasetParamChange}
                  />
                ))}
              </div>
              <Button className="mt-4" onClick={generateDataset}>Generate Dataset</Button>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="model">
          <Card>
            <CardHeader>Train and Select Model</CardHeader>
            <CardContent>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {models.map(model => (
                    <SelectItem key={model} value={model}>{model}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <div className="mt-4">
                <Button className="mr-2" onClick={trainModel}>Train Model</Button>
                <Button onClick={selectModel}>Select Model</Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="history">
          <Card>
            <CardHeader>Prediction History</CardHeader>
            <CardContent>
              <ul>
                {history.map((item, index) => (
                  <li key={index} className="mb-2">
                    {item.transaction}: {item.prediction}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="audit">
          <Card>
            <CardHeader>Audit Performance</CardHeader>
            <CardContent>
              <Button onClick={auditPerformance}>Audit Performance</Button>
              {auditResults && (
                <div className="mt-4">
                  <p>False Positive Rate: {auditResults.false_positive_rate}</p>
                  <p>False Negative Rate: {auditResults.false_negative_rate}</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SecureBankDashboard;