# EM Clustering Algorithm for Unsupervised Classification of Articles

## General

In this exercise, you will implement the EM clustering algorithm for unsupervised classification of articles into clusters. The task involves clustering articles based on their content (unigrams) while utilizing topic information only for evaluation purposes. Refer to the document "Underflow Scaling and Smoothing in EM" for detailed instructions.

## Input

The input for this task is the `develop.txt` file, which contains articles for clustering. Each article's header includes topic labels, which are used for evaluation but ignored during clustering.

## Unsupervised Classification

### Goal
Your goal is to cluster articles into 9 clusters using the EM algorithm. Each article will be assigned to the cluster with the highest likelihood.

## Implementation Instructions

### Underflow

To manage underflow issues, follow the guidelines provided in the document "Underflow Scaling and Smoothing in EM".

### Time and Space Complexity

Filter out rare words (occurring 3 times or less in the corpus) to optimize time and space complexity.

### Smoothing

Experiment with different values of lambda (λ) in the M-step smoothing to achieve optimal performance.

### EM Initialization

Initialize the EM algorithm by splitting the 2124 articles in `develop.txt` into 9 initial clusters in a modulo-9 manner.

## Report

### Stopping Criterion

Define your stopping criterion for the EM iterations and report it.

### Log Likelihood and Perplexity

Print the log likelihood after each iteration and plot a graph where:
- X-axis: Iteration number
- Y-axis: Log likelihood

Calculate and plot the mean perplexity per word using the formula:
\[
\text{Perplexity} = e^{-\frac{\text{Log Likelihood}}{N}}
\]
where \( N \) is the number of words in the dataset.

### Confusion Matrix

Create a 9x9 confusion matrix where:
- Rows represent clusters
- Columns represent topics from `topics.txt`
- Include an additional column for cluster sizes (number of articles assigned)

Sort rows by cluster size in descending order and present the confusion matrix in a readable format.

### Histograms

Use Excel or a similar tool to create 9 histograms (one per cluster) where:
- X-axis: Topics (in the same order as in the confusion matrix columns)
- Y-axis: Number of articles from each topic in the cluster

Label each histogram with its dominant topic (topic with the most articles).

### Accuracy

Calculate and report the accuracy of your classification:
\[
\text{Accuracy} = \frac{\text{Number of correct assignments}}{\text{Total number of assignments}}
\]

Ensure your model achieves an accuracy greater than 0.6.

### Constants and Parameters

Report the following:
- \( k \) (from underflow treatment)
- Vocabulary size after filtering (should be 6,800)
- Smoothing parameter \( \lambda \) used in the M-step.



