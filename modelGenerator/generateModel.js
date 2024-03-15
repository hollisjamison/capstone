const fs = require("fs");
const csv = require("csv-parser");
const tf = require("@tensorflow/tfjs-node");

function createVocabulary(reviews) {
  const vocabulary = new Set();
  reviews.forEach((review) => {
    review.forEach((word) => {
      vocabulary.add(word);
    });
  });
  return vocabulary;
}

function tokenizeReviews(reviews) {
  return reviews.map((review) => review.split(" "));
}

function tokenizeAndMapReviews(reviews, wordIndex) {
  return reviews.map((review) =>
    review.split(" ").map((word) => wordIndex[word] || 0)
  );
}

function convertSentimentToLabel(sentiment) {
  if (sentiment === "positive") return 2;
  if (sentiment === "neutral") return 1;
  if (sentiment === "negative") return 0;
}

function padSequences(tokenizedReviews) {
  const maxLength = tokenizedReviews.reduce(
    (max, arr) => Math.max(max, arr.length),
    0
  );

  const paddedReviews = tokenizedReviews.map((arr) => {
    const pad = Array(maxLength - arr.length).fill(0);
    return arr.concat(pad);
  });

  return { paddedReviews, maxLength };
}

function compileModel(model) {
  model.compile({
    optimizer: "adam",
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"],
  });
}

function createModel(vocabSize, inputLength) {
  const model = tf.sequential();
  model.add(
    tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: 50,
      inputLength: inputLength,
    })
  );
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  return model;
}

async function trainModel(paddedReviewsTensor, sentimentLabels, vocabSize, inputLength) {
  const model = createModel(vocabSize, inputLength);
  compileModel(model);

  await model.fit(paddedReviewsTensor, sentimentLabels, {
    batchSize: 32,
    epochs: 10,
    validationSplit: 0.2,
  });

  await model.save('file://./sentiment_model');
}

module.exports = {
  trainModel,
  createModel,
  compileModel,
  padSequences,
  convertSentimentToLabel,
  tokenizeAndMapReviews,
  tokenizeReviews,
  createVocabulary
};