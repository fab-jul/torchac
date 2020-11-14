# MNIST AutoEncoder

<div align="center">
  <img src='progress_plot.png' width="100%"/>
</div>

This example shows how to train a simple auto-encoder for MNIST,
with a quantized bottleneck, and a conditional probability
model that estimates the distribution of the bottleneck. It basically
follows recent image compression papers in terms of structure,
but is much simpler and not tuned.

Goals:
- Show how to use the probability distribution predicted from some probability
model CNN to estimate the bitrate
- Actually encode symbols with that model using `torchac`, and see how many
bits it takes to store

What could be improved:
- The models don't actually train very well. The purpose of the example is to
show how `torchac` can be used.

## Runnig it

The example supports interactive plots with matplotlib, which it tries
to use by default. If they are not available (in headless environments),
it falls back to storing plots in `plots`. If you want to force
non-interactive, you can set `export NO_INTERACTIVE=1` before
running the script.

```bash
pip install matplotlib  # If not installed already
cd examples/mnist_qutoencder
python mnist_autoencoder_example.py
```
