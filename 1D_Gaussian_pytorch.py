# 1D Gaussian distribution approximation using GAN
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os


# Data distribution
class DataDistribution:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self, num_samples):
        samples = np.random.normal(self.mu, self.sigma, num_samples)
        samples.sort()
        return samples


# Noise distribution
class NoiseDistribution:
    def __init__(self, data_range):
        self.data_range = data_range

    def sample(self, num_samples):
        offset = np.random.random(num_samples) * 0.01
        samples = np.linspace(-self.data_range, self.data_range, num_samples) + offset
        return samples


# Generator model
class Generator(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # Fully-connected layer
        fc = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        # initializer
        torch.nn.init.normal(fc.weight)
        torch.nn.init.constant(fc.bias, 0.0)

        # Hidden layer
        self.hidden_layer = torch.nn.Sequential(
            fc,
            torch.nn.ReLU()
        )

        # Output layer
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim, bias=True)
        # initializer
        torch.nn.init.normal(self.output_layer.weight)
        torch.nn.init.constant(self.output_layer.bias, 0.0)

    def forward(self, x):
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out


# Discriminator
class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # Fully-connected layer
        fc1 = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        # initializer
        torch.nn.init.normal(fc1.weight)
        torch.nn.init.constant(fc1.bias, 0.0)

        # Hidden layer
        self.hidden_layer = torch.nn.Sequential(
            fc1,
            torch.nn.ReLU()
        )

        # Fully-connected layer
        fc2 = torch.nn.Linear(hidden_dim, output_dim, bias=True)
        # initializer
        torch.nn.init.normal(fc2.weight)
        torch.nn.init.constant(fc2.bias, 0.0)

        # Output layer
        self.output_layer = torch.nn.Sequential(
            fc2,
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out


# Test samples
class TestSample:
    def __init__(self, discriminator, generator, data, gen, data_range, batch_size, num_samples, num_bins):
        self.D = discriminator
        self.G = generator
        self.data = data
        self.gen = gen
        self.B = batch_size
        self.num_samples = num_samples
        self.num_bins = num_bins
        self.xs = np.linspace(-data_range, data_range, num_samples)
        self.bins = np.linspace(-data_range, data_range, num_bins)

    def decision_boundary(self):
        db = np.zeros((self.num_samples, 1))
        for i in range(self.num_samples // self.B):
            x_ = self.xs[self.B*i:self.B*(i+1)]
            x_ = Variable(torch.FloatTensor(np.reshape(x_, [self.B, 1])))

            db[self.B*i:self.B*(i+1)] = self.D(x_).data.numpy()

        return db

    def data_distribution(self):
        d = self.data.sample(self.num_samples)
        p_data, _ = np.histogram(d, self.bins, density=True)
        return p_data

    def gen_distribution(self):
        zs = self.xs
        # zs = self.gen.sample(num_samples)
        g = np.zeros((self.num_samples, 1))
        for i in range(self.num_samples // self.B):
            z_ = zs[self.B * i:self.B * (i + 1)]
            z_ = Variable(torch.FloatTensor(np.reshape(z_, [self.B, 1])))

            g[self.B * i:self.B * (i + 1)] = self.G(z_).data.numpy()

        p_gen, _ = np.histogram(g, self.bins, density=True)
        return p_gen


# Display result
class Display:
    def __init__(self, num_samples, num_bins, data_range, mu, sigma):
        self.num_samples = num_samples
        self.num_bins = num_bins
        self.data_range = data_range
        self.mu = mu
        self.sigma = sigma


    def plot_result(self, db_pre_trained, db_trained, p_data, p_gen):
        d_x = np.linspace(-self.data_range, self.data_range, len(db_trained))
        p_x = np.linspace(-self.data_range, self.data_range, len(p_data))

        f, ax = plt.subplots(1)
        ax.plot(d_x, db_pre_trained, '--', label='Decision boundary(pre-trained)')
        ax.plot(d_x, db_trained, label='Decision boundary')
        ax.set_ylim(0, max(1, np.max(p_data) * 1.1))
        ax.set_xlim(max(self.mu - self.sigma * 3, -self.data_range * 0.9), min(self.mu + self.sigma * 3, self.data_range * 0.9))
        plt.plot(p_x, p_data, label='Real data')
        plt.plot(p_x, p_gen, label='Generated data')
        plt.title('1D Gaussian Approximation using vanilla GAN: ' + '(mu: %3g,' % self.mu + ' sigma: %3g)' % self.sigma)
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend(loc=1)
        plt.grid(True)

        # Save plot
        save_dir = "results/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        plt.savefig(save_dir + '1D_Gaussian' + '_mu_%g' % self.mu + '_sigma_%g' % self.sigma + '.png')

        plt.show()

# Parameters
mu = 1.0
sigma = 1.5
data_range = 5
batch_size = 150

input_dim = 1
hidden_dim = 32
output_dim = 1
num_epochs = 3000
num_epochs_pre = 1000
learning_rate = 0.03

# Samples
data = DataDistribution(mu, sigma)
gen = NoiseDistribution(data_range)

# Models
G = Generator(input_dim, hidden_dim, output_dim)
D = Discriminator(input_dim, hidden_dim, output_dim)

# Loss function
criterion = torch.nn.BCELoss()


# Pre-training discriminator
# optimizer
optimizer = torch.optim.SGD(D.parameters(), lr=learning_rate)

D_pre_losses = []
num_samples_pre = 1000
num_bins_pre = 100
for epoch in range(num_epochs_pre):
    # Generate samples
    d = data.sample(num_samples_pre)
    histc, edges = np.histogram(d, num_bins_pre, density=True)

    # Estimate pdf
    max_histc = np.max(histc)
    min_histc = np.min(histc)
    y_ = (histc - min_histc) / (max_histc - min_histc)
    x_ = edges[1:]

    x_ = Variable(torch.FloatTensor(np.reshape(x_, [num_bins_pre, input_dim])))
    y_ = Variable(torch.FloatTensor(np.reshape(y_, [num_bins_pre, output_dim])))

    # Train model
    optimizer.zero_grad()
    D_pre_decision = D(x_)
    D_pre_loss = criterion(D_pre_decision, y_)
    D_pre_loss.backward()
    optimizer.step()

    # Save loss values for plot
    D_pre_losses.append(D_pre_loss[0].data.numpy())

    if epoch % 100 == 0:
        print(epoch, D_pre_loss.data.numpy())

# Plot loss
fig, ax = plt.subplots()
losses = np.array(D_pre_losses)
plt.plot(losses, label='Pre-train loss')
plt.title("Pre-training Loss")
plt.legend()
plt.show()

# Test sample after pre-training
num_samples = 10000
num_bins = 20
sample = TestSample(D, G, data, gen, data_range, batch_size, num_samples, num_bins)

db_pre_trained = sample.decision_boundary()


# Training GAN
# Optimizers
D_optimizer = torch.optim.SGD(D.parameters(), lr=learning_rate)
G_optimizer = torch.optim.SGD(G.parameters(), lr=learning_rate)

D_losses = []
G_losses = []
for epoch in range(num_epochs):
    # Generate samples
    x_ = data.sample(batch_size)
    x_ = Variable(torch.FloatTensor(np.reshape(x_, [batch_size, input_dim])))
    y_real_ = Variable(torch.ones([batch_size, output_dim]))
    y_fake_ = Variable(torch.zeros([batch_size, output_dim]))

    # Train discriminator with real data
    D_real_decision = D(x_)
    D_real_loss = criterion(D_real_decision, y_real_)

    # Train discriminator with fake data
    z_ = gen.sample(batch_size)
    z_ = Variable(torch.FloatTensor(np.reshape(z_, [batch_size, input_dim])))
    z_ = G(z_)

    D_fake_decision = D(z_)
    D_fake_loss = criterion(D_fake_decision, y_fake_)

    # Back propagation
    D_loss = D_real_loss + D_fake_loss
    D.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    # Train generator
    z_ = gen.sample(batch_size)
    z_ = Variable(torch.FloatTensor(np.reshape(z_, [batch_size, input_dim])))
    z_ = G(z_)

    D_fake_decision = D(z_)
    G_loss = criterion(D_fake_decision, y_real_)

    # Back propagation
    D.zero_grad()
    G.zero_grad()
    G_loss.backward()
    G_optimizer.step()

    # Save loss values for plot
    D_losses.append(D_loss[0].data.numpy())
    G_losses.append(G_loss[0].data.numpy())

    if epoch % 100 == 0:
        print(epoch, D_loss.data.numpy(), G_loss.data.numpy())

# Test sample after pre-training
sample = TestSample(D, G, data, gen, data_range, batch_size, num_samples, num_bins)

db_trained = sample.decision_boundary()
p_data = sample.data_distribution()
p_gen = sample.gen_distribution()

# Plot losses
fig, ax = plt.subplots()
D_losses = np.array(D_losses)
G_losses = np.array(G_losses)
plt.plot(D_losses, label='Discriminator')
plt.plot(G_losses, label='Generator')
plt.title("Training Losses")
plt.legend()
plt.show()

# Display result
display = Display(num_samples, num_bins, data_range, mu, sigma)
display.plot_result(db_pre_trained, db_trained, p_data, p_gen)