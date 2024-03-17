import React, { useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [review, setReview] = useState('');
  const [prediction, setPrediction] = useState('');

  const loadModel = async () => {
    const model = await tf.loadLayersModel('/sentiment_model/model.json');
    return model;
  };

  const predictReview = async () => {
    const model = await loadModel();
    
    const preprocessedReview = review.trim().toLowerCase();
    const reviewArray = Array.from(preprocessedReview);
    const reviewLength = reviewArray.length;
    const paddedReview = reviewArray.slice(0, 571).concat(Array(571 - reviewLength).fill(' '));
    const inputData = paddedReview.map((char) => char.charCodeAt(0));
    
    const input = tf.tensor2d([inputData], [1, 571]);
    const prediction = model.predict(input);
    const classIndex = prediction.argMax(1).dataSync()[0];
    const classLabels = ['Negative', 'Neutral', 'Positive'];
    prediction.data().then((predictionValues) => {
      console.log(predictionValues);
    });
    setPrediction(classLabels[classIndex]);
  };

  return (
    <div className="container mt-5">
      <h1>Review Classifier</h1>
      <div className="form-group">
        <label htmlFor="reviewInput">Enter a review:</label>
        <textarea
          className="form-control"
          id="reviewInput"
          rows="3"
          value={review}
          onChange={(e) => setReview(e.target.value)}
        ></textarea>
      </div>
      <button className="btn btn-primary" onClick={predictReview}>
        Predict
      </button>
      {prediction && (
        <div className="mt-3">
          <h3>Prediction: {prediction}</h3>
        </div>
      )}
    </div>
  );
}

export default App;