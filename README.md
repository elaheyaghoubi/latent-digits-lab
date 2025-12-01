# latent-digits-lab

Experiments with convolutional neural networks (CNNs) and convolutional autoencoders on the MNIST handwritten digits dataset.  
The focus is on understanding how architecture and training hyperparameters affect:

- Classification performance (train / test accuracy)
- Training time
- Reconstruction quality in an autoencoder setting

## Project goals

- Train a CNN classifier for MNIST digits with different hyperparameter configurations.
- Measure how choices like learning rate, number of epochs, and channel sizes impact:
  - Training accuracy
  - Test accuracy
  - Training time
- Build a convolutional autoencoder that:
  - Learns a compact latent representation of digits
  - Reconstructs input images with minimal error
- Compare different autoencoder architectures and training setups using:
  - Reconstruction loss
  - Visual inspection of reconstructed samples
  - (Optionally) the same metrics used for the classifier

## Tech stack

- Python 3.x  
- [PyTorch](https://pytorch.org/)  
- [torchvision](https://pytorch.org/vision/stable/index.html)  
- NumPy  
- Matplotlib  
- pandas  
- Jupyter Notebook

## Repository structure

```text
.
├── main.ipynb        # All experiments: data loading, CNN, autoencoder, evaluation
└── data/             # MNIST data will be downloaded here automatically
```

The entire workflow lives in main.ipynb so you can follow the experiments from top to bottom.

## Dataset

The code uses the standard MNIST dataset from torchvision.datasets.MNIST:

- 60,000 training images
- 10,000 test images
- Grayscale, 28×28 pixels
- Labels: digits 0–9

The dataset is downloaded automatically on first run.

## Setup

1. Clone the repository
```
git clone https://github.com/<your-username>/latent-digits-lab.git
cd latent-digits-lab
```
2. Create and activate a virtual environment (optional but recommended)
```
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

```
3. Install dependencies
```
pip install torch torchvision matplotlib numpy pandas jupyter
```
Then open ```main.ipynb```.
How to run the experiments

1. Open main.ipynb in Jupyter.

2. Run the cells from top to bottom in order.

3. For each experiment block:

- Review the model_configs list that defines the architectures / hyperparameters.

- Execute the training loop cell.

- Inspect:

  - Printed logs for training / test accuracy and training time

  - The resulting pandas DataFrame summarizing all runs
 
## CNN classifier

The notebook defines:

- A convolutional network (several conv layers + nonlinearity + pooling + fully connected layers)

- A training loop that:

  - Trains on the MNIST training set

  - Evaluates on the test set

  - Measures:

    - Training accuracy

    - Test accuracy

    - Training time (per configuration)

The ```model_configs``` list controls variations such as:

- Number of channels in each conv layer

- Activation function

- Learning rate

- Number of epochs

Results are collected into a list of dictionaries and converted into a pandas DataFrame for easier comparison.

## Convolutional autoencoder

The notebook also defines a ConvAutoencoder class with:

- An encoder built from convolutional layers

- A decoder built from transpose-convolution layers

- Configurable:

  - Channel sizes

  - Activation function (e.g., ReLU, LeakyReLU, Tanh)

  - Learning rate

  - Number of epochs

The training loop:

- Minimizes mean squared error (MSE) between input and reconstructed images

- Logs reconstruction loss per epoch

- Stores summary statistics (e.g., loss, training time) in a results list that is again converted to a DataFrame

You can also visualize original vs reconstructed images by adding a small plotting cell if desired.
Interpreting results

## Interpreting results

Typical analyses you can do with the notebook output:

- Compare configurations by:

  - Test accuracy vs training time for the classifier

  - Reconstruction loss vs training time for the autoencoder

- Check for:

  - Underfitting (low train and test accuracy)

  - Overfitting (high train accuracy but noticeably lower test accuracy)

- Inspect reconstructed digits to see how:

  - Deeper encoders change reconstruction quality

  - Different activations affect smoothness and sharpness

## Customization ideas

Some ways to extend this project:

- Add dropout or batch normalization to the CNN and observe the impact.

- Try different optimizers (SGD with momentum, AdamW, etc.).

- Log metrics with tools like TensorBoard or Weights & Biases.

- Swap MNIST with Fashion-MNIST to see how the models generalize to a slightly harder dataset.

- Experiment with different latent-space sizes in the autoencoder and measure the trade-off between compression and reconstruction quality.
