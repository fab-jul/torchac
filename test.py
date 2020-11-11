import pytest
import torch
import torchac


def test_out_of_range_symbol():
  cdf_float = torch.tensor([0., 1/3, 2/3, 1.], dtype=torch.float32).reshape(1, -1)
  assert list(_encode_decode(cdf_float, [10],
                             needs_normalization=False,
                             check_input_bounds=False)) == [False]


def test_uniform_float():
  cdf_float = torch.tensor([0., 1/3, 2/3, 1.], dtype=torch.float32).reshape(1, -1)

  # Check if integer conversion works as expected.
  cdf_int = torchac._convert_to_int_and_normalize(cdf_float,
                                                  needs_normalization=False)
  assert cdf_int[0, 1] == 2**16//3
  assert cdf_int[0, -1] == 0

  # Check if we can uniquely encode without normalization.
  assert all(_encode_decode(cdf_float,
                            symbols_to_check=(0, 1, 2),
                            needs_normalization=False))


def test_uniform_float_multipdim():
  cdf_float = torch.tensor([0., 1/3, 2/3, 1.], dtype=torch.float32).reshape(1, -1)

  L = 3
  C, H, W = 5, 8, 9

  Lp = L + 1
  cdf_float = torch.cat([cdf_float for _ in range(C*H*W)], dim=0)
  cdf_float = cdf_float.reshape(C, H, W, -1)
  assert cdf_float.shape[-1] == Lp

  sym = torch.arange(C * H * W, dtype=torch.int16) % L
  sym = sym.reshape(C, H, W)

  byte_stream = torchac.encode_float_cdf(
    cdf_float,
    sym,
    needs_normalization=False,
    check_input_bounds=True)
  sym_out = torchac.decode_float_cdf(
    cdf_float,
    byte_stream,
    needs_normalization=False)
  assert sym_out.equal(sym)


def test_normalize_float():
  #  Two times the same value -> needs to be normalized!
  cdf_float = torch.tensor([0., 1/3, 1/3, 1.], dtype=torch.float32).reshape(1, -1)
  # Check if we can uniquely encode
  assert all(_encode_decode(cdf_float,
                            symbols_to_check=(0, 1, 2),
                            needs_normalization=True))

  # Should raise because symbol is out of bounds.
  with pytest.raises(ValueError):
    sym = torch.tensor([3], dtype=torch.int16)
    torchac.encode_float_cdf(cdf_float, sym,
                             needs_normalization=True,
                             check_input_bounds=True)


def test_normalization_sigmoid():
  mu = 0
  L = 256
  Lp = L + 1
  x_for_cdf = torch.linspace(-1, 1, Lp)
  # Logistic distribution.
  for sigma in [0.001, 0.01, 0.1, 1., 10.]:
    cdf_float = torch.sigmoid((x_for_cdf-mu)/sigma)

    # Put it into the expected shape.
    cdf_float = cdf_float.reshape(1, -1)

    # Check if we can uniquely decode all valid symbols.
    assert all(_encode_decode(
      cdf_float, symbols_to_check=range(L), needs_normalization=True))


def _encode_decode(cdf_float, symbols_to_check,
                   needs_normalization, check_input_bounds=True):
  # Check if we can uniquely encode
  for symbol in symbols_to_check:
    sym = torch.tensor([symbol], dtype=torch.int16)
    byte_stream = torchac.encode_float_cdf(
      cdf_float,
      sym,
      needs_normalization=needs_normalization,
      check_input_bounds=check_input_bounds)
    sym_out = torchac.decode_float_cdf(
      cdf_float,
      byte_stream,
      needs_normalization=needs_normalization)
    yield sym_out == sym
