import pytest
import numpy as np
from main import april

def test_april():
    # Dummy data for testing
    # These are not realistic values, but are useful for checking the shapes
    # and the basic computation.
    d_model = 512
    d_k = 64
    d_v = 64

    # Dummy weights
    w = {
        'q': np.random.rand(d_model, d_k),
        'k': np.random.rand(d_model, d_k),
        'v': np.random.rand(d_model, d_v)
    }

    # Dummy gradients
    dldw = {
        'q': np.random.rand(d_model, d_k),
        'k': np.random.rand(d_model, d_k),
        'v': np.random.rand(d_model, d_v)
    }

    # Dummy loss derivative
    dldz = np.random.rand(1, d_model) # Assuming batch size of 1

    # Dummy attention function (not used in the current version of april)
    F = None

    # Run the APRIL algorithm
    z = april(F, w, dldw, dldz)

    # Check the shape of the output
    assert z.shape == (d_model, 1)

if __name__ == "__main__":
    pytest.main()
