import dataclasses
import itertools
import warnings

import os
import time

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import torchvision
import matplotlib.pyplot as plt
import matplotlib

try:
  import torchac
except ImportError:
  raise ImportError('torchac is not available! Please see the main README '
                    'on how to install it.')


# Set to true to write to disk.
_WRITE_BITS = False


# Interactive plots setup
_FORCE_NO_INTERACTIVE_PLOTS = int(os.environ.get('NO_INTERACTIVE', 0)) == 1


if not _FORCE_NO_INTERACTIVE_PLOTS:
  try:
    matplotlib.use("TkAgg")
    interactive_plots_available = True
  except ImportError:
    warnings.warn(f'*** TkAgg not available! Saving plots...')
    interactive_plots_available = False
else:
  interactive_plots_available = False

if not interactive_plots_available:
  matplotlib.use("Agg")


# Set seed.
torch.manual_seed(0)


def train_test_loop(bottleneck_size=2,
                    L=5,
                    batch_size=8,
                    lr=1e-2,
                    rate_loss_enable_itr=500,
                    num_test_batches=10,
                    train_plot_every_itr=50,
                    mnist_download_dir='data',
                    ):
  """Train and test an autoencoder.

  :param bottleneck_size: Number of channels in the bottleneck.
  :param L: Number of levels that we quantize to.
  :param batch_size: Batch size we train with.
  :param lr: Learning rate of Adam.
  :param rate_loss_enable_itr: Iteration when the rate loss is enabled.
  :param num_test_batches: Number of batches we test on (randomly chosen).
  :param train_plot_every_itr: How often to update the train plot.
  :param mnist_download_dir: Where to store MNIST.
  """
  ae = Autoencoder(bottleneck_size, L)
  prob = ConditionalProbabilityModel(L=L, bottleneck_shape=ae.bottleneck_shape)
  mse = nn.MSELoss()
  adam = torch.optim.Adam(
    itertools.chain(ae.parameters(), prob.parameters()),
    lr=lr)

  train_acc = Accumulator()
  test_acc = Accumulator()
  plotter = Plotter()

  transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(
      # Make images 32x32.
      lambda image: F.pad(image, pad=(2, 2, 2, 2), mode='constant'))
  ])

  train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(mnist_download_dir, train=True, download=True,
                               transform=transforms),
    batch_size=batch_size, shuffle=True)

  rate_loss_enabled = False
  for i, (images, labels) in enumerate(train_loader):
    assert images.shape[-2:] == (32, 32)

    adam.zero_grad()

    # Get reconstructions and symbols from the autoencoder.
    reconstructions, sym = ae(images)
    assert sym.shape[1:] == ae.bottleneck_shape

    # Get estimated and real bitrate from probability model, given labels.
    bits_estimated, bits_real = prob(sym.detach(), labels)
    mse_loss = mse(reconstructions, images)

    # If we are beyond iteration `rate_loss_enable_itr`, enable a rate loss.
    if i < rate_loss_enable_itr:
      loss = mse_loss
    else:
      loss = mse_loss + 1/1000 * bits_estimated
      rate_loss_enabled = True

    loss.backward()
    adam.step()

    # Update Train Plot.
    if i > 0 and i % train_plot_every_itr == 0:
      train_acc.append(i, bits_estimated, bits_real, mse_loss)
      print(f'{i: 10d}; '
            f'loss={loss:.3f}, '
            f'bits_estimated={bits_estimated:.3f}, '
            f'mse={mse_loss:.3f}')
      plotter.update('Train',
                     images, reconstructions, sym, train_acc, rate_loss_enabled)

    # Update Test Plot.
    if i > 0 and i % 100 == 0:
      print(f'{i: 10d} Testing on {num_test_batches} random batches...')
      ae.eval()
      prob.eval()
      test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
          mnist_download_dir, train=False, download=True, transform=transforms),
        batch_size=batch_size, shuffle=True)
      with torch.no_grad():
        across_batch_acc = Accumulator()
        for j, (test_images, test_labels) in enumerate(test_loader):
          if j >= num_test_batches:
            break
          test_reconstructions, test_sym = ae(test_images)
          test_bits_estimated, test_bits_real = prob(test_sym, test_labels)
          test_mse_loss = mse(test_reconstructions, test_images)
          across_batch_acc.append(j, test_bits_estimated, test_bits_real, test_mse_loss)
        test_bits_estimated_mean, test_bits_real_mean, test_mse_loss_mean = \
            across_batch_acc.means()
        test_acc.append(
          i, test_bits_estimated_mean, test_bits_real_mean, test_mse_loss_mean)
        plotter.update('Test', test_images, test_reconstructions, test_sym,
                       test_acc, rate_loss_enabled)
      ae.train()
      prob.train()


class STEQuantize(torch.autograd.Function):
  """Straight-Through Estimator for Quantization.

  Forward pass implements quantization by rounding to integers,
  backward pass is set to gradients of the identity function.
  """
  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    return x.round()

  @staticmethod
  def backward(ctx, grad_outputs):
    return grad_outputs


class Autoencoder(nn.Module):
  def __init__(self, bottleneck_size, L):
    if L % 2 != 1:
      raise ValueError(f'number of levels L={L}, must be odd number!')
    super(Autoencoder, self).__init__()
    self.L = L
    self.enc = nn.Sequential(
      nn.Conv2d(1, 32, 5, stride=2, padding=2),
      nn.InstanceNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 32, 5, stride=2, padding=2),
      nn.InstanceNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 32, 5, stride=2, padding=2),
      nn.InstanceNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 32, 5, stride=2, padding=2),
      nn.InstanceNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, bottleneck_size, 1, stride=1, padding=0, bias=False),
    )

    self.dec = nn.Sequential(
      nn.ConvTranspose2d(bottleneck_size, 32, 5, stride=2, padding=2, output_padding=1),
      nn.InstanceNorm2d(32),
      nn.ReLU(),
      nn.ConvTranspose2d(32, 32, 5, stride=2, padding=2, output_padding=1),
      nn.InstanceNorm2d(32),
      nn.ReLU(),
      nn.ConvTranspose2d(32, 32, 5, stride=2, padding=2, output_padding=1),
      nn.InstanceNorm2d(32),
      nn.ReLU(),
      nn.ConvTranspose2d(32, 32, 5, stride=2, padding=2, output_padding=1),
      nn.InstanceNorm2d(32),
      nn.ReLU(),

      # Add a few convolutions at the final resolution.
      nn.Conv2d(32, 32, 3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 32, 1, stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(32, 1, 1, stride=1),
    )

    self.quantize = STEQuantize.apply
    self.bottleneck_shape = (bottleneck_size, 2, 2)

  def forward(self, image):
    # Encode image x into the latent.
    latent = self.enc(image)
    # The jiggle is there so that the lowest and highest symbol are not at
    # the boundary. Probably not needed.
    jiggle = 0.2
    spread = self.L - 1 + jiggle
    # The sigmoid clamps to [0, 1], then we multiply it by spread and substract
    # spread / 2, so that the output is nicely centered around zero and
    # in the interval [-spread/2, spread/2]
    latent = torch.sigmoid(latent) * spread - spread / 2
    latent_quantized = self.quantize(latent)
    reconstructions = self.dec(latent_quantized)
    sym = latent_quantized + self.L // 2
    return reconstructions, sym.to(torch.long)


class ConditionalProbabilityModel(nn.Module):
  def __init__(self, L, bottleneck_shape):
    super(ConditionalProbabilityModel, self).__init__()
    self.L = L
    self.bottleneck_shape = bottleneck_shape

    self.bottleneck_size, _, _ = bottleneck_shape

    # We predict a value for each channel, for each level.
    num_output_channels = self.bottleneck_size * L

    self.model = nn.Sequential(
      nn.Conv2d(1, self.bottleneck_size, 3, padding=1),
      nn.BatchNorm2d(self.bottleneck_size),
      nn.ReLU(),
      nn.Conv2d(self.bottleneck_size, self.bottleneck_size, 3, padding=1),
      nn.BatchNorm2d(self.bottleneck_size),
      nn.ReLU(),
      nn.Conv2d(self.bottleneck_size, num_output_channels, 1, padding=0),
    )

  def forward(self, sym, labels):
    batch_size = sym.shape[0]
    _, H, W = self.bottleneck_shape
    # Construct the input, which is just the label of the current number
    # at each spatial location.
    bottleneck_shape_with_batch_dim = (batch_size, 1, H, W)
    static_input = torch.ones(
      bottleneck_shape_with_batch_dim, dtype=torch.float32)
    dynamic_input = static_input * labels.reshape(-1, 1, 1, 1)
    # Divide by 9 and substract 0.5 to center the input around 0 and make
    # it be contained in [-0.5, 0.5].
    dynamic_input = dynamic_input / 9 - 0.5

    # Get the output of the CNN.
    output = self.model(dynamic_input)
    _, C, H, W = output.shape
    assert C == self.bottleneck_size * self.L

    # Reshape it such that the probability per symbol has it's own dimension.
    # output_reshaped has shape (B, C, L, H, W).
    output_reshaped = output.reshape(
      batch_size, self.bottleneck_size, self.L, H, W)
    # Take the softmax over that dimension to make this into a normalized
    # probability distribution.
    output_probabilities = F.softmax(output_reshaped, dim=2)
    # Permute the symbols dimension to the end, as expected by torchac.
    # output_probabilities has shape (B, C, H, W, L).
    output_probabilities = output_probabilities.permute(0, 1, 3, 4, 2)
    # Estimate the bitrate from the PMF.
    estimated_bits = estimate_bitrate_from_pmf(output_probabilities, sym=sym)
    # Convert to a torchac-compatible CDF.
    output_cdf = pmf_to_cdf(output_probabilities)
    # Get real bitrate from the byte_stream.
    sym = sym.to(torch.int16)
    byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True)
    real_bits = len(byte_stream) * 8
    if _WRITE_BITS:
      # Write to a file.
      with open('outfile.b', 'wb') as fout:
        fout.write(byte_stream)
      # Read from a file.
      with open('outfile.b', 'rb') as fin:
        byte_stream = fin.read()
    assert torchac.decode_float_cdf(output_cdf, byte_stream).equal(sym)
    return estimated_bits, real_bits


def pmf_to_cdf(pmf):
  cdf = pmf.cumsum(dim=-1)
  spatial_dimensions = pmf.shape[:-1] + (1,)
  zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
  cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
  return cdf_with_0


def estimate_bitrate_from_pmf(pmf, sym):
  L = pmf.shape[-1]
  pmf = pmf.reshape(-1, L)
  sym = sym.reshape(-1, 1)
  assert pmf.shape[0] == sym.shape[0]
  relevant_probabilities = torch.gather(pmf, dim=1, index=sym)
  bitrate = torch.sum(-torch.log2(relevant_probabilities.clamp(min=1e-3)))
  return bitrate


@dataclasses.dataclass
class Accumulator:
  def __init__(self):
    self.iterations = []
    self.bits_estimated_acc = []
    self.bits_real_acc = []
    self.mse_acc = []

  def append(self, i, bits_estimated, bits_real, mse):
    self.iterations.append(i)
    self.bits_estimated_acc.append(bits_estimated.item())
    self.bits_real_acc.append(bits_real)
    self.mse_acc.append(mse.item())

  def get_errors(self):
    return [real / estimated - 1 for real, estimated in
            zip(self.bits_real_acc, self.bits_estimated_acc)]

  def means(self):
    return (np.mean(self.bits_estimated_acc),
            np.mean(self.bits_real_acc),
            np.mean(self.mse_acc))


class Plotter(object):
  def __init__(self):
    plt.ion()
    self.fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(12, 8))
    Plotter._setup_axes(axs[:, :2], 'Train')
    Plotter._setup_axes(axs[:, 2:], 'Test')
    plt.tight_layout()
    if interactive_plots_available:
      plt.draw()
    else:
      unique_id = str(time.time()).replace('.', '_')
      self.out_dir = os.path.join('plots', unique_id)
      os.makedirs(self.out_dir, exist_ok=True)
    self.axs = axs

  @staticmethod
  def _setup_axes(axs, title):
    axs[0, 0].set_axis_off()
    axs[0, 0].set_title(f'{title} Input')
    axs[0, 1].set_axis_off()
    axs[0, 1].set_title(f'{title} Reconstruction')
    axs[1, 0].set_axis_off()
    axs[1, 0].set_title(f'{title} Bottleneck channel 1')
    axs[1, 1].set_axis_off()
    axs[1, 1].set_title(f'{title} Bottleneck channel 2')
    axs[2, 0].set_title(f'Estimated Bitrate {title}')
    axs[2, 1].set_title(f'Real Bitrate {title}')
    axs[3, 0].set_title(f'MSE {title}')
    axs[3, 1].set_title('Rel. Bitrate Error')

  def update(self,
             mode,
             images, reconstructions, sym,
             acc: Accumulator,
             rate_loss_enabled: bool):
    # First two columns are for training, second two for testing.
    axs = self.axs[:, :2] if mode == 'Train' else self.axs[:, 2:]
    title = mode

    # Plot images.
    axs[0, 0].imshow(images[0, ...].squeeze())
    axs[0, 1].imshow(reconstructions[0, ...].squeeze().detach().numpy())
    axs[1, 0].imshow(sym[0, 0, ...].to(torch.float).detach().numpy())
    axs[1, 1].imshow(sym[0, 1, ...].to(torch.float).detach().numpy())

    # Plot lines, make sure to empty plot first.
    axs[2, 0].clear()
    axs[2, 1].clear()
    axs[3, 0].clear()
    axs[3, 1].clear()
    axs[2, 0].set_title(f'Estimated Bitrate {title}')
    axs[2, 1].set_title(f'Real Bitrate {title}')
    axs[3, 0].set_title(f'MSE {title}')
    axs[3, 1].set_title('Rel. Bitrate Error')

    linestyle = '-' if rate_loss_enabled else ':'
    color = 'b' if mode == 'Train' else 'r'
    linestyle = color + linestyle
    axs[2, 0].plot(
      acc.iterations, acc.bits_estimated_acc, linestyle)
    axs[2, 1].plot(
      acc.iterations, acc.bits_real_acc, linestyle)

    axs[3, 0].plot(
      acc.iterations, acc.mse_acc, color)
    axs[3, 1].plot(
      acc.iterations, acc.get_errors(), color)

    if interactive_plots_available:
      plt.pause(0.05)
    else:
      plotname = f'{acc.iterations[-1]:010d}.png'
      out_p = os.path.join(self.out_dir, plotname)
      print(f'Saving plot at {out_p}...')
      self.fig.savefig(out_p)


if __name__ == '__main__':
  train_test_loop()
