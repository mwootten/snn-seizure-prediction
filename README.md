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

The multi-spiking model we tried has a remarkably poor performance. The single-spiking network (which was created by some of our collaborators) did slightly better, but not better than a simple convolutional network, or best of all, an SVM trained on the final layer.

## Running the code

First, complete the first of the preprocessing steps. You can find the instructions for [downloading data](https://github.com/matthewlw/snn-seizure-prediction/wiki/Retrieving-Raw-Data) and [running the Fourier transform](https://github.com/matthewlw/snn-seizure-prediction/wiki/FFT) on our project wiki.

After that, you should be able to use the Makefile in the project root. Create a folder named `build`, then move the Makefile into that directory; this should make the relative paths work out correctly. It has dependencies on `awk`, `jq`, and PyPy. Please be advised that we have not attempted to run this code on Windows; while there's no clear reason why it should fail, it will almost certainly require a lot of tweaking. Run `make spiking-predicted` and `make spiking-sans2-predicted` to run the test predictions for the multi-spiking network, along with all the intermediate steps. `make spiking-true` and `make spiking-sans2-true` should create all the true test labels. These files correspond line-by-line; that is, if the 10th line of the 'true' file reads 1, then the 10th line of the 'predicted' file should have a positive prediction (though a mistake here is an incorrect prediction of the model, not a bug in the code). 

An output of 15 signals a positive prediction; an output of 18 signals a negative one. Therefore, straightforward sorting methods for calculating the ROC have to flip the sign, or else get a worse-than-chance result. An output that is neither 15 nor 18 should be interpreted as an uncertain prediction of whichever bound is closer.

The single-spiking code is in the `single-spiking` folder. I don't have documentation on its function, other than knowing it's based off Lee, Delbruck, and Pfeiffer (2016) -- described on the project wiki -- and that it has the same inputs and outputs as the multi-spiking network.

## Authors

- Multi-spiking code: Jeremy Angel and Matthew Wootten (Loudoun Academy of Science)
- SIngle-spiking code: Jimin Chae and Junwon Kim (Daegu Science High School)
