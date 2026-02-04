import React, { useState } from "react";

export default function Home() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [activeTab, setActiveTab] = useState("prediction");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handlePredict = () => {
    if (!file) {
      alert("Please upload a file first.");
      return;
    }
    // Dummy prediction logic
    setPrediction("Predicted Genre: Hindustani Classical");
    setExplanation(
      "Explanation: The model focused on rhythmic patterns and pitch distribution common in Hindustani Classical music."
    );
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", justifyContent: "center", alignItems: "center", background: "#f8f9fa" }}>
      <div style={{ background: "white", padding: "40px", borderRadius: "12px", boxShadow: "0px 4px 10px rgba(0,0,0,0.1)", width: "80%" }}>
        <h1 style={{ textAlign: "center", marginBottom: "20px" }}>
          Indian Music Genre Classification
        </h1>

        <input
          type="file"
          accept=".mp3,.wav"
          onChange={handleFileChange}
          style={{ marginBottom: "10px", width: "100%" }}
        />

        <button
          onClick={handlePredict}
          style={{
            width: "100%",
            padding: "10px",
            backgroundColor: "#007bff",
            color: "white",
            border: "none",
            borderRadius: "6px",
            cursor: "pointer",
            marginBottom: "20px"
          }}
        >
          Get Prediction
        </button>

        {/* Tab Switcher */}
        <div style={{ display: "flex", justifyContent: "space-around", marginBottom: "10px" }}>
          <button
            onClick={() => setActiveTab("prediction")}
            style={{
              flex: 1,
              padding: "8px",
              background: activeTab === "prediction" ? "#007bff" : "#e9ecef",
              color: activeTab === "prediction" ? "white" : "black",
              border: "none",
              borderRadius: "6px 0 0 6px",
              cursor: "pointer"
            }}
          >
            Prediction
          </button>
          <button
            onClick={() => setActiveTab("explain")}
            style={{
              flex: 1,
              padding: "8px",
              background: activeTab === "explain" ? "#007bff" : "#e9ecef",
              color: activeTab === "explain" ? "white" : "black",
              border: "none",
              borderRadius: "0 6px 6px 0",
              cursor: "pointer"
            }}
          >
            Explain Prediction
          </button>
        </div>

        {/* Tab Content */}
        <div>
          {activeTab === "prediction" ? (
            prediction ? (
              <p style={{ color: "green", fontWeight: "bold" }}>{prediction}</p>
            ) : (
              <p style={{ color: "gray" }}>Upload a file and click Predict.</p>
            )
          ) : explanation ? (
            <p>{explanation}</p>
          ) : (
            <p style={{ color: "gray" }}>No explanation yet.</p>
          )}
        </div>
      </div>
    </div>
  );
}
