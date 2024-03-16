const tf = require("@tensorflow/tfjs-node-gpu");
const fs = require("fs");
const readline = require("readline");

const { tokenizeAndMapReviews } = require('./generateModel');

function padSequences(tokenizedReview, maxLength) {
  const paddedReview = tokenizedReview.slice(0, maxLength);
  const padding = Array(maxLength - paddedReview.length).fill(0);
  const paddedSequence = [paddedReview.concat(padding)];
  return paddedSequence;
}

function convertLabelToSentiment(label) {
  const sentimentLabels = ['negative', 'neutral', 'positive'];
  return sentimentLabels[label];
}

const init = async () => {
  const loadedModel = await tf.loadLayersModel(
    "file://./sentiment_model/model.json"
  );

  const maxLength = JSON.parse(fs.readFileSync('./sentiment_model/maxLength.json'));
  const wordIndex = JSON.parse(fs.readFileSync('./sentiment_model/wordIndex.json'));

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  rl.question("Enter your review: ", (reviewText) => {
    const newReview = [reviewText];
    const newTokenizedReview = tokenizeAndMapReviews(newReview, wordIndex)[0];
    const newPaddedReview = padSequences(newTokenizedReview, maxLength);
    const newPaddedReviewTensor = tf.tensor2d(
      newPaddedReview,
      [1, maxLength],
      "float32"
    );

    const prediction = loadedModel.predict(newPaddedReviewTensor);
    const sentimentClass = prediction.argMax(-1).dataSync()[0];
    const sentimentLabel = convertLabelToSentiment(sentimentClass);
    console.log("Predicted sentiment:", sentimentLabel);

    rl.close();
  });
};

init();