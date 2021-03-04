import numpy as np, pandas as pd
import argparse


class BonzLayer:
    def __init__(self):
        self.lr = 0.001

    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        pass


class BonzLinear(BonzLayer):
    def __init__(self, input_dim, output_dim, bias=True):
        """
        Linear(X) = XW + B
        W: (i, o)
        X: (b, i)
        B: (o)
        """
        super(BonzLinear, self).__init__()
        self.weight = np.random.normal(scale=0.01, size=(input_dim, output_dim))
        if bias:
            self.bias = np.zeros(output_dim)

    def forward(self, inputs):
        return np.dot(inputs, self.weight) + self.bias

    def backward(self, inputs, previous_grad):
        grad_layer = np.dot(previous_grad, self.weight.T)

        # compute gradient w.r.t. weights and biases
        grad_weight = np.dot(inputs.T, previous_grad)
        grad_bias = previous_grad.mean(axis=0) * inputs.shape[0]

        # Here we perform a stochastic gradient descent step.
        self.weight -= self.lr * grad_weight
        self.bias -= self.lr * grad_bias

        return grad_layer


class BonzBCEwSigmoid(BonzLayer):
    def __init__(self):
        super(BonzBCEwSigmoid, self).__init__()

    def sigmoid(self, inputs):
        return 1 / (1 + np.exp(1)**-inputs)

    def forward(self, predict, true):
        self.y = true
        self.z = self.sigmoid(predict) # Calculate sigmoid
        loss = - self.y*np.log(self.z) - (1-self.y)*np.log(1-self.z)
        return loss.mean()

    def backward(self):
        # (z-y) / [batch * (z - z(.)z)]
        # return (self.z - self.y) / (self.z - np.multiply(self.z, self.z))
        return self.z - self.y


class BonzModel():
    def __init__(self, num_layer=2, layer_dims=[128, 1], lr=1e-3):
        assert len(layer_dims) == num_layer and layer_dims[-1] == 1
        layer_dims = [54] + layer_dims
        self.layers = [
            BonzLinear(input_dim=layer_dims[i], output_dim=layer_dims[i+1]) for i in range(len(layer_dims)-1)
        ]
        self.loss_fn = BonzBCEwSigmoid()
        self.lr = lr

    def create_batches(self, shuffle=True, train=True):
        n = self.train_data.shape[0] if train else self.test_data.shape[0]
        num_batch = n / self.batch_size if n % self.batch_size == 0 else n // self.batch_size + 1

        indexes = np.arange(n)
        if shuffle:
            np.random.shuffle(indexes)

        return np.array_split(indexes, num_batch)

    def set_lr(self):
        for layer in self.layers:
            layer.lr = self.lr

    def training_step(self):
        for epoch in range(self.epoch):
            loss_batch = [] # Store loss every batch

            # Load train data & label by batch
            for index in self.create_batches(shuffle=True):
                inputs = self.train_data[index]
                labels = self.train_label[index].reshape(-1,1)
                # Forward path
                output_arr = [inputs]
                for layer in self.layers:
                    output = layer(output_arr[-1])
                    output_arr.append(output)

                loss = self.loss_fn(output_arr[-1], labels)
                loss_batch.append(loss)

                # Backward path
                output_arr.pop()
                previous_grad = self.loss_fn.backward()  # The first loss comes from the loss function
                for layer in self.layers[::-1]:
                    previous_grad = layer.backward(output_arr.pop(), previous_grad)

            avg_loss = np.array(loss_batch).mean()
            print(f'Loss at epoch={epoch}: {avg_loss}')

    def testing_step(self):
        predicts = [] # Store loss every batch

        # Load train data & label by batch
        for index in self.create_batches(shuffle=False, train=False):
            inputs = self.test_data[index]
            # Forward path
            for layer in self.layers:
                inputs = layer(inputs)
            predicts.extend(inputs.squeeze().tolist())

        return (sigmoid(np.array(predicts)) > 0.5).astype(np.int)

    def train(self, training_data: np.ndarray, epoch=50, batch_size=32):
        self.train_data = training_data[:, 1:].astype(np.float)
        self.train_label = training_data[:, 0].astype(np.float)
        self.epoch = epoch
        self.batch_size = batch_size

        self.set_lr()
        self.training_step()

    def predict(self, test_data: np.ndarray):
        self.test_data = test_data
        return self.testing_step()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_layer', default=2)
    parser.add_argument('--layer_dims', default=[128,1])
    parser.add_argument('--lr', default=1e-3)

    return parser.parse_args()


def sigmoid(inputs):
    return 1 / (1 + np.exp(1)**-inputs)


if __name__ == '__main__':
    args = get_args() # I like argparse for a easy debugging

    # Load data
    training_data = np.loadtxt(open('data/testing_spam.csv'), delimiter=',').astype(np.int)

    # Init model
    classifier = BonzModel(num_layer=args.num_layer, layer_dims=args.layer_dims)

    # Train & test model
    classifier.train(training_data)

    testing_spam = np.loadtxt(open('data/testing_spam.csv'), delimiter=',').astype(np.int)
    test_data = testing_spam[:, 1:]
    test_labels = testing_spam[:, 0]

    predictions = classifier.predict(test_data)
    accuracy = np.count_nonzero(predictions == test_labels) / test_labels.shape[0]
    print(f"Accuracy on test data is: {accuracy}")

