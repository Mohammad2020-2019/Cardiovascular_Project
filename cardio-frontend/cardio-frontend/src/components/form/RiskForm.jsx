import React, { useState } from "react";
import axios from "axios";

const RiskForm = () => {
  const [formData, setFormData] = useState({
    Age: 50,
    Sex: "M",           // 0 = Male, 1 = Female
    ChestPainType: "ATA", // TA, ATA, NAP, ASY
    Cholesterol: 200.0,
    FastingBS: 0,     // 0 = <120 mg/dl, 1 = >120 mg/dl
    MaxHR: 150,
    ExerciseAngina: "N", // 0 = No, 1 = Yes
    Oldpeak: 1.0,
    ST_Slope: "Flat"  // Up, Flat, Down
  });

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleInputChange = (field, value) => {
    setFormData({ ...formData, [field]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    try {
      const response = await axios.post(
          "http://localhost:8000/Heart_Failure_predict",
          formData
      );
      setResult(response.data);
    } catch (error) {
      setError("Error calculating risk. Please try again.");
      console.error("API Error:", error);
    }
  };



  return (
      <div className="form-container">
        <h2>Heart Disease Risk Assessment</h2>
        <form onSubmit={handleSubmit}>
          {/* Age */}
          <div className="form-group">
            <label>Age (years):</label>
            <input
                type="number"
                value={formData.Age}
                onChange={(e) => handleInputChange('Age', parseInt(e.target.value))}
                min="0"
                max="120"
                required
            />
          </div>

          {/* Sex */}
          <div className="form-group">
            <label>Sex:</label>
            <select
                value={formData.Sex}
                onChange={(e) => handleInputChange('Sex', e.target.value)}
            >
              <option value="M">Male</option>
              <option value="F">Female</option>
            </select>
          </div>


          {/* Chest Pain Type */}
          <div className="form-group">
            <label>Chest Pain Type:</label>
            <select
                value={formData.ChestPainType}
                onChange={(e) => handleInputChange('ChestPainType', e.target.value)}
            >
              <option value="TA">Typical Angina</option>
              <option value="ATA">Atypical Angina</option>
              <option value="NAP">Non-Anginal Pain</option>
              <option value="ASY">Asymptomatic</option>
            </select>
          </div>

          {/* Cholesterol */}
          <div className="form-group">
            <label>Cholesterol (mg/dl):</label>
            <input
                type="number"
                value={formData.Cholesterol}
                onChange={(e) => handleInputChange('Cholesterol', parseFloat(e.target.value))}
                min="0"
                max="600"
                step="1"
                required
            />
          </div>

          {/* Fasting Blood Sugar */}
          <div className="form-group">
            <label>Fasting Blood Sugar:</label>
            <select
                value={formData.FastingBS}
                onChange={(e) => handleInputChange('FastingBS', parseInt(e.target.value))}
            >
              <option value={0}>Normal (&lt; 120 mg/dl)</option>
              <option value={1}>High (&gt; 120 mg/dl)</option>
            </select>
          </div>

          {/* Max Heart Rate */}
          <div className="form-group">
            <label>Max Heart Rate:</label>
            <input
                type="number"
                value={formData.MaxHR}
                onChange={(e) => handleInputChange('MaxHR', parseInt(e.target.value))}
                min="60"
                max="220"
                required
            />
          </div>

          {/* Exercise Angina */}
          <div className="form-group">
            <label>Exercise-Induced Angina:</label>
            <select
                value={formData.ExerciseAngina}
                onChange={(e) => handleInputChange('ExerciseAngina', e.target.value)}
            >
              <option value="N">No</option>
              <option value="Y">Yes</option>
            </select>
          </div>

          {/* Oldpeak */}
          <div className="form-group">
            <label>ST Depression (Oldpeak):</label>
            <input
                type="number"
                value={formData.Oldpeak}
                onChange={(e) => handleInputChange('Oldpeak', parseFloat(e.target.value))}
                step="0.1"
                min="0"
                max="10"
                required
            />
          </div>

          {/* ST Slope */}
          <div className="form-group">
            <label>ST Slope:</label>
            <select
                value={formData.ST_Slope}
                onChange={(e) => handleInputChange('ST_Slope', e.target.value)}
            >
              <option value="Up">Upsloping</option>
              <option value="Flat">Flat</option>
              <option value="Down">Downsloping</option>
            </select>
          </div>

          <button type="submit">Assess Heart Risk</button>
        </form>

        {result && (
            <div className={`result ${result.diagnosis === "Heart Disease" ? "heart-disease" : "no-heart-disease"}`}>
              <h3>Diagnosis: {result.diagnosis}</h3>
              <p>Probability: {result.probability}%</p>
            </div>
        )}

        {error && <div className="error-message">{error}</div>}
      </div>
  );
};

export default RiskForm;