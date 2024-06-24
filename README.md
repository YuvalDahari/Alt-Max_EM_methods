
# EM Clustering Algorithm for Unsupervised Classification of Articles

In this assignment, we applied the EM clustering algorithm to categorize a collection of articles into nine predefined topics. Here are the key findings and analyses from our implementation:

## Threshold to Stop the EM Iterations

We set the stopping criterion (`epsilon`) for the EM algorithm to 0.01. This means that if the change in likelihood between iterations falls below 0.01, we consider the algorithm to have converged and stop further iterations.


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

## Threshold to Stop the EM Iterations

We set the stopping criterion (`epsilon`) for the EM algorithm to 0.01. This means that if the change in likelihood between iterations falls below 0.01, we consider the algorithm to have converged and stop further iterations.

## Log Likelihood & Perplexity Graphs

We plotted two graphs to monitor the algorithm's progress:
- **Log Likelihood**: This graph shows the increase in log likelihood with each iteration, demonstrating that the algorithm is converging towards a solution.
- **Perplexity**: The perplexity graph depicts how the perplexity decreases or remains stable across iterations. Lower perplexity indicates better model performance in predicting article topics.

## Confusion Matrix (9x9)

The confusion matrix illustrates the correspondence between predicted clusters and actual topics. Rows represent clusters sorted by size, and columns represent topics from our dataset. Each cell \( M_{ij} \) indicates the number of articles classified into cluster \( i \) that belong to topic \( j \).

## Histograms for Confusion Matrix

We created histograms corresponding to each cluster, showing the distribution of articles across topics within that cluster. These histograms provide a visual representation of which topics are dominant within each cluster.

## Accuracy

The accuracy of our model was calculated to be approximately 61.5%. This metric reflects the percentage of correctly classified articles compared to the total number of articles processed.
\[
\text{Accuracy} = \frac{\text{Number of correct assignments}}{\text{Total number of assignments}}
\]

## More Hyperparameters and Parameters

### Vocabulary Size After Filtering

After filtering out rare words (those appearing three times or less), our vocabulary size reduced to 6,800 unique words. This filtering step helps improve computational efficiency and model accuracy.

### Smoothing Parameter (\( \lambda \))

We experimented with different values and selected \( \lambda = 0.023 \) for Lidstone smoothing during the M-step. This parameter adjustment was crucial given our large vocabulary size (300,000 words), ensuring that all words contribute meaningfully to topic probability calculations.

### Handling Underflow (Parameter \( k \))

To mitigate underflow issues during calculations, we employed a scaling factor \( k \). This technique involved logarithmic transformation and normalization of probabilities to prevent excessively large or small values, thereby stabilizing the algorithm's computations.
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




