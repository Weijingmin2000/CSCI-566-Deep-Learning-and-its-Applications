from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.layer_utils import *
from lib.optim import *
from tqdm import tqdm

class DataLoader(object):
    """
    Data loader class.

    Arguments:
    - data: Array of input data, of shape (batch_size, d_1, ..., d_k)
    - labels: Array of labels, of shape (batch_size,)
    - batch_size: The size of each returned minibatch
    """
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.indices = np.asarray(range(data.shape[0]))

    # reset the indices to be full length
    def _reset(self):
        self.indices = np.asarray(range(self.data.shape[0]))

    # Call this shuffle function after the last batch for each epoch
    def _shuffle(self):
        np.random.shuffle(self.indices)

    # Get the next batch of data
    def get_batch(self):
        if len(self.indices) < self.batch_size:
            self._reset()
            self._shuffle()
        indices_curr = self.indices[0:self.batch_size]
        data_batch = self.data[indices_curr]
        labels_batch = self.labels[indices_curr]
        self.indices = np.delete(self.indices, range(self.batch_size))
        return data_batch, labels_batch


def compute_acc(model, data, labels, num_samples=None, batch_size=100):
    """
    Compute the accuracy of given data and labels

    Arguments:
    - data: Array of input data, of shape (batch_size, d_1, ..., d_k)
    - labels: Array of labels, of shape (batch_size,)
    - num_samples: If not None, subsample the data and only test the model
      on these sampled datapoints.
    - batch_size: Split data and labels into batches of this size to avoid using
      too much memory.

    Returns:
    - accuracy: Scalar indicating fraction of inputs that were correctly
      classified by the model.
    """
    N = data.shape[0]
    if num_samples is not None and N > num_samples:
        indices = np.random.choice(N, num_samples)
        N = num_samples
        data = data[indices]
        labels = labels[indices]

    num_batches = N // batch_size
    if N % batch_size != 0:
        num_batches += 1
    preds = []
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        output = model.forward(data[start:end], False)
        scores = softmax(output)
        pred = np.argmax(scores, axis=1)
        preds.append(pred)
    preds = np.hstack(preds)
    accuracy = np.mean(preds == labels)
    return accuracy


""" Some comments """
def train_net(data, model, loss_func, optimizer, batch_size, max_epochs,
              lr_decay=1.0, lr_decay_every=1000, show_every=10, verbose=False,
              regularization="none", reg_lambda=0.0):
    """
    Train a network with this function, parameters of the network are updated
    using stochastic gradient descent methods defined in optim.py.

    The parameters which achive the best performance after training for given epochs
    will be returned as a param dict. The training history and the validation history
    is returned for post analysis.

    Arguments:
    - data: Data instance should look like the followings:
    - data_dict = {
        "data_train": (# Training data,   # Training GT Labels),
        "data_val":   (# Validation data, # Validation GT Labels),
        "data_test":  (# Testing data,    # Testing GT Labels),
      }
    - model: An instance defined in the fully_conn.py, with a sequential object as attribute
    - loss_func: An instance defined in the layer_utils.py, we only introduce cross-entropy
      classification loss for this part of assignment
    - batch_size: Batch size of the input data
    - max_epochs: The total number of epochs to train the model
    - lr_decay: The amount to decay the learning rate
    - lr_decay_every: Decay the learning rate every given epochs
    - show_every: Show the training information every given iterations
    - verbose: To show the information or not
    - regularization: Which regularization method to use: "l1", "l2". Default: "none"
    - reg_lambda: paramter that controls the strength of regularization. Decault: 0.0

    Returns:
    - opt_params: optimal parameters
    - loss_hist: Loss recorded during training
    - train_acc_hist: Training accuracy recorded during training
    - val_acc_hist: Validation accuracy recorded during training
    """

    # Initialize the variables
    data_train, labels_train = data["data_train"]
    data_val, labels_val = data["data_val"]
    dataloader = DataLoader(data_train, labels_train, batch_size)
    opt_val_acc = 0.0
    opt_params = None
    loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    # Compute the maximum iterations and iterations per epoch
    iters_per_epoch = int(max(data_train.shape[0] / batch_size, 1))
    max_iters = int(iters_per_epoch  * max_epochs)

    # Start the training
    for epoch in range(max_epochs):
        # Compute the starting iteration and ending iteration for current epoch
        iter_start = epoch * iters_per_epoch
        iter_end   = (epoch + 1) * iters_per_epoch

        # Decay the learning rate every specified epochs
        if epoch % lr_decay_every == 0 and epoch > 0:
            optimizer.lr = optimizer.lr * lr_decay
            print ("Decaying learning rate of the optimizer to {}".format(optimizer.lr))

        # Main training loop
        for iter in tqdm(range(iter_start, iter_end)):
            data_batch, labels_batch = dataloader.get_batch()
            # print(data_batch.shape, labels_batch.shape)
            #############################################################################
            # TODO: Update the parameters by a forward pass for the network, a backward #
            # pass to the network, and make a step for the optimizer.                   #
            # Notice: In backward pass, you should enable regularization.               #
            # Store the loss to loss_hist                                               #
            #############################################################################
            output = model.forward(data_batch)
            loss = loss_func.forward(output, labels_batch) # loss
            dloss = loss_func.backward() # cross_entropy backward
            grads = model.backward(dloss, regularization, reg_lambda) # loss backward, record gradient
            optimizer.step() # update weight using gradient
            loss_hist.append(loss) # loss recorded
            #############################################################################
            #                             END OF YOUR CODE                              #
            #############################################################################

            # Show the training loss
            if verbose and iter % show_every == 0:
                last_losses = loss_hist[-show_every:]
                avg_loss = sum(last_losses)/len(last_losses)
                print ("(Iteration {} / {}) Average loss: {}".format(iter+1, max_iters, avg_loss))

        # End of epoch, compute the accuracies
        train_acc = 0
        val_acc = 0
        #############################################################################
        # TODO: Compute the training accuracy and validation accuracy using         #
        # compute_acc method, store the results to train_acc and val_acc,           #
        # respectively                                                              #
        #############################################################################
        train_acc = compute_acc(model, data_train, labels_train, None, batch_size)
        val_acc = compute_acc(model, data_val, labels_val, None, batch_size)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        # Save the best params for the model
        if val_acc > opt_val_acc:
            #############################################################################
            # TODO: Save the optimal parameters to opt_params variable by name using    #
            # model.net.gather_params method                                            #
            #############################################################################
            if opt_params is None:
              opt_params = {}
            model.net.gather_params()
            for n, v in model.net.params.items():
                opt_params[n] = v # process the parameters
            opt_val_acc = val_acc
            #############################################################################
            #                             END OF YOUR CODE                              #
            #############################################################################

        # Show the training accuracies
        if verbose:
            print ("(Epoch {} / {}) Training Accuracy: {}, Validation Accuracy: {}".format(
            epoch+1, max_epochs, train_acc, val_acc))

    return opt_params, loss_hist, train_acc_hist, val_acc_hist
