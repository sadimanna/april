import pytest
import torch
from models.custom_vit import CustomBlock

def test_custom_block_instantiation():
    """
    Tests that CustomBlock can be instantiated without errors.
    This is the fix for the TypeError that was occurring.
    """
    try:
        CustomBlock(dim=768, num_heads=12)
    except TypeError:
        pytest.fail("TypeError was raised during CustomBlock instantiation.")

if __name__ == "__main__":
    pytest.main()
