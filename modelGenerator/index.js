const fs = require("fs");
const csv = require("csv-parser");
const tf = require("@tensorflow/tfjs-node");
const {
  trainModel,
  createModel,
  compileModel,
  padSequences,
  convertSentimentToLabel,
  tokenizeAndMapReviews,
  tokenizeReviews,
  createVocabulary
} = require('./generateModel');

const reviews = [];
const sentiments = [];

fs.createReadStream("./data/reviewTrainingData.csv")
  .pipe(csv())
  .on("data", (row) => {
    reviews.push(row.cleaned_review);
    sentiments.push(convertSentimentToLabel(row.sentiments));
  })
  .on("end", () => {
    const tokenizedReviews = tokenizeReviews(reviews);
    const sentimentLabels = tf.tensor1d(sentiments, 'float32');
    const vocabulary = createVocabulary(tokenizedReviews);
    const vocabSize = vocabulary.size + 1;
    const wordIndex = {};
    let index = 1;
    vocabulary.forEach((word) => {
      wordIndex[word] = index++;
    });
    const mappedReviews = tokenizeAndMapReviews(reviews, wordIndex);
    const { paddedReviews, maxLength } = padSequences(mappedReviews);
    const paddedReviewsTensor = tf.tensor2d(paddedReviews, [paddedReviews.length, maxLength], 'float32');

    fs.writeFileSync('./sentiment_model/maxLength.json', JSON.stringify(maxLength));
    fs.writeFileSync('./sentiment_model/wordIndex.json', JSON.stringify(wordIndex));
    trainModel(paddedReviewsTensor, sentimentLabels, vocabSize, maxLength);
  });