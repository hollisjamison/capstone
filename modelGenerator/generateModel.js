const fs = require("fs");
const csv = require("csv-parser");
const tf = require("@tensorflow/tfjs-node-gpu");

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
      outputDim: 45,
      maskZero: true,
      inputLength: inputLength,
    })
  );

  model.add(
    tf.layers.lstm({
      units: 100,
    })
  );

  model.add(
    tf.layers.dropout({
      rate: 0.2,
    })
  );

  model.add(
    tf.layers.dense({
      units: 100,
      activation: "relu",
    })
  );

  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
    })
  );

  model.add(
    tf.layers.dense({
      units: 32,
      activation: "relu",
    })
  );

  model.add(
    tf.layers.dense({
      units: 3,
      activation: "softmax",
    })
  );

  return model;
}

async function trainModel(
  paddedReviewsTensor,
  sentimentLabels,
  vocabSize,
  inputLength
) {
  const model = createModel(vocabSize, inputLength);
  compileModel(model);

  const logs = "./logs";
  const tensorBoardCallback = tf.node.tensorBoard(logs, {
    updateFreq: "epoch",
    histogramFreq: 1,
  });

  await model.fit(paddedReviewsTensor, sentimentLabels, {
    batchSize: 32,
    epochs: 10,
    validationSplit: 0.2,
    callbacks: [tensorBoardCallback],
  });

  await model.save("file://./sentiment_model");
}

module.exports = {
  trainModel,
  createModel,
  compileModel,
  padSequences,
  convertSentimentToLabel,
  tokenizeAndMapReviews,
  tokenizeReviews,
  createVocabulary,
};
