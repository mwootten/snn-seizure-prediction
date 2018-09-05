# Multiple-spiking neural networks for seizure prediction
We seek to predict seizures from EEGs using two models of spiking neural networks.

![Overall data flow diagram](https://cdn.rawgit.com/matthewlw/snn-seizure-prediction/ae4c1623/diagrams/uncolored-flowchart.svg)

## Input

We took training data from the IEEG Dataset "Study 005". This was an 8 day recording from a 21-year old male patient with simple partial seizures. The recording contains a number of channels from both sides of the brain; we selected LTD4 as our sole channel to analyze. This was motivated primarily by a desire to reduce the data input size.

## Feature selection

We decided to use the frequency-space representation of the brain waves, because it allowed us to more effectively pick out important classes of brainwaves, and to reduce the input data volume while reducing the amount of meaningful data as much as possible. In the 8-30 Hz range, we selected 135 samples which were (approximately) uniformly spaced in frequency space.

## Models

We started with a CNN with layers 135 > 45 > 15 > 5 > 1. From this, we stripped off the first and second to last layers and used these as input to a multi-spiking neural network (MuSpiNN, as described in Ghosh-Dastidar and Adeli (2009)) as well as an RBF-kernel SVM (using the scikit-learn defaults). The multi-spiking neural network with input from the last CNN hidden layer had layer sizes 21 > 5 > 1, and the network with input from the second to last CNN hidden layer had layer sizes 61 > 15 > 4 > 1. The apparent mismatch between successive layer sizes (5 to 21 and 15 to 61) comes from the transformation used to convert the real-valued intermediates from the CNN to the discrete inputs to the multi-spiking network (four spiking inputs per output), plus a single bias neuron. We chose the successive layer sizes for the MuSpiNN were created by trying to keep the ratio between layers relatively constant. 

## Evaluation

Since we framed this as a binary classification problem, we decided to use binary classification metrics to assess our performance. Since we artificially made this problem have balanced classes, we were able to use sensitivity-specificity (ROC) curves, rather than precision-recall. 

![Multiple ROC plot](https://cdn.rawgit.com/matthewlw/snn-seizure-prediction/ae4c1623/diagrams/final-roc.svg)

The multi-spiking model we tried has a remarkably poor performance. The single-spiking network (which was created by some of our collaborators, and whose code is not on our repository) did slightly better, but not better than a simple convolutional network, or best of all, an SVM trained on the final layer.

## Running the code

TODO: add further instructions

## Authors

- Primary: Jeremy Angel and Matthew Wootten (Loudoun Academy of Science)
- Other contributors: Jimin Chae and Junwon Kim (Daegu Science High School)
