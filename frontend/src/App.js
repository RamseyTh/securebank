import React from 'react';
import SecureBankDashboard from './SecureBankDashboard';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>SecureBank Fraud Detection System</h1>
      </header>
      <main>
        <SecureBankDashboard />
      </main>
      <footer>
        <p>© 2024 SecureBank. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;