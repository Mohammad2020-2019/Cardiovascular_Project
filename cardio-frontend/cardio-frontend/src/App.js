// src/App.js
import React, { useState } from 'react';
import RiskForm from './components/form/RiskForm.jsx';
import './App.css';

function App() {
  const [formRisk, setFormRisk] = useState(null);
  const [chatRisk, setChatRisk] = useState(null);

  return (
      <div className="app-container">
        <h1>Cardiovascular Risk Assessment</h1>

        <div className="interface-container">
          {/* Form-based Input */}
          <div className="form-section">
            <h2>Traditional Form</h2>
            <RiskForm onResult={(risk) => setFormRisk(risk)} />
          </div>


        </div>

      </div>
  );
}

export default App;