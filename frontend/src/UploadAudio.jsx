import React, { useState, useRef } from "react";
import axios from "axios";
import "./UploadAudio.css";

const UploadAudio = () => {
  const [file, setFile] = useState(null);
  const [activeTab, setActiveTab] = useState("predict");
  const [prediction, setPrediction] = useState([]);
  const [limeHtml, setLimeHtml] = useState("");
  const [shapImg, setShapImg] = useState("");
  const [permBar, setPermBar] = useState("");
  const [loading, setLoading] = useState(false);
  const [audioURL, setAudioURL] = useState(null);

  const audioRef = useRef(null);
  const fileInputRef = useRef(null);

  // Handle file change
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setPrediction([]);
    setLimeHtml("");
    setShapImg("");
    setPermBar("");
    setAudioURL(selectedFile ? URL.createObjectURL(selectedFile) : null);
  };

  // Clear selected file
  const handleClearFile = () => {
    setFile(null);
    setAudioURL(null);
    setPrediction([]);
    setLimeHtml("");
    setShapImg("");
    setPermBar("");
    if (fileInputRef.current) fileInputRef.current.value = null;
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  };

  // Upload file and call backend endpoint
  const uploadAndPredict = async (endpoint, callback) => {
    if (!file && endpoint !== "explain/permutation")
      return alert("Select a file first");
    setLoading(true);

    const formData = new FormData();
    if (file) formData.append("file", file);

    try {
      const res = await axios.post(`http://localhost:5000/${endpoint}`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      callback(res.data);
    } catch (err) {
      console.error(err);
      alert("Error fetching from backend");
    } finally {
      setLoading(false);
    }
  };

  // Actions
  const handlePredict = () =>
    uploadAndPredict("predict", (data) => setPrediction(data.ann_top3));

  const handleLime = () => {
    setLimeHtml("");
    uploadAndPredict("explain/lime", (data) => setLimeHtml(data.lime_html));
  };

  const handleShap = () =>
    uploadAndPredict("explain/shap", (data) => setShapImg(data.shap_image));

  const handlePermutation = () => {
    setPermBar("");
    uploadAndPredict("explain/permutation", (data) => setPermBar(data.bar_plot));
  };

  // Download helper
  const downloadBase64Image = (base64, filename) => {
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${base64}`;
    link.download = filename;
    link.click();
  };

  return (
    <div className="upload-container">
      <h2 className="upload-title">🎵 Music Genre Classification & xAI</h2>

      {/* File Upload Section */}
      <div className="file-upload">
        <input
          ref={fileInputRef}
          type="file"
          accept=".wav,.mp3"
          onChange={handleFileChange}
        />
      </div>

      {file && audioURL && (
        <div className="audio-section">
          <audio ref={audioRef} controls src={audioURL} />
          <button className="clear-btn" onClick={handleClearFile}>
            Clear File
          </button>
        </div>
      )}

      {/* Tab Buttons */}
      <div className="tab-buttons">
        <button
          className={activeTab === "predict" ? "active" : ""}
          onClick={() => setActiveTab("predict")}
        >
          Predict Genre
        </button>
        <button
          className={activeTab === "lime" ? "active" : ""}
          onClick={() => setActiveTab("lime")}
        >
          LIME
        </button>
        <button
          className={activeTab === "shap" ? "active" : ""}
          onClick={() => setActiveTab("shap")}
        >
          SHAP
        </button>
        <button
          className={activeTab === "permutation" ? "active" : ""}
          onClick={() => setActiveTab("permutation")}
        >
          Permutation
        </button>
        <button
          className={activeTab === "about" ? "active" : ""}
          onClick={() => setActiveTab("about")}
        >
          About
        </button>
      </div>

      {loading && <p className="loading-text">Loading...</p>}

      {/* Prediction Section */}
      {activeTab === "predict" && (
        <div className="tab-content">
          <button className="action-btn" onClick={handlePredict}>
            Predict
          </button>
          {prediction.length > 0 && !loading && (
            <div className="results-section">
              <h3>Confidence Scores</h3>
              <ol>
                {prediction.map((p, idx) => (
                  <li key={idx} className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${p.probability * 100}%` }}
                    >
                      {p.genre} — {(p.probability * 100).toFixed(2)}%
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}

      {/* LIME Section */}
      {activeTab === "lime" && (
        <div className="tab-content">
          <button className="action-btn" onClick={handleLime}>
            Get LIME Explanation
          </button>
          {limeHtml && (
            <>
              <iframe
                title="LIME Explanation"
                srcDoc={limeHtml}
                className="lime-frame"
              />
              <button
                className="small-btn"
                onClick={() =>
                  navigator.clipboard
                    .writeText(limeHtml)
                    .then(() => alert("Copied to clipboard"))
                }
              >
                Copy LIME HTML
              </button>
            </>
          )}
        </div>
      )}

      {/* SHAP Section */}
      {activeTab === "shap" && (
        <div className="tab-content">
          <button className="action-btn" onClick={handleShap}>
            Get SHAP Explanation
          </button>
          {shapImg && (
            <div className="image-container">
              <img
                src={`data:image/png;base64,${shapImg}`}
                alt="SHAP Plot"
                className="explain-img"
              />
              <button
                className="small-btn"
                onClick={() => downloadBase64Image(shapImg, "shap_plot.png")}
              >
                Download SHAP Plot
              </button>
            </div>
          )}
        </div>
      )}

      {/* Permutation Section */}
      {activeTab === "permutation" && (
        <div className="tab-content">
          <button className="action-btn" onClick={handlePermutation}>
            Compute Permutation Importance
          </button>
          {permBar && (
            <div className="image-container">
              <h4>Feature Importance</h4>
              <img
                src={`data:image/png;base64,${permBar}`}
                alt="Permutation Bar Plot"
                className="explain-img"
              />
              <button
                className="small-btn"
                onClick={() =>
                  downloadBase64Image(permBar, "permutation_plot.png")
                }
              >
                Download Permutation Plot
              </button>
            </div>
          )}
        </div>
      )}

      {/* About Section */}
      {activeTab === "about" && (
        <div className="tab-content about-section">
          <h4>Learn about MFCC</h4>
          <a
            href="https://en.wikipedia.org/wiki/Mel-frequency_cepstrum"
            target="_blank"
            rel="noreferrer"
          >
            Mel-frequency cepstrum (Wikipedia)
          </a>
        </div>
      )}
    </div>
  );
};

export default UploadAudio;
