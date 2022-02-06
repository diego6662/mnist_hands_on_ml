import pytest
from src.main.mnist_chapter4 import load_dataset, split_dataset


def test_load_dataset():
    X, y = load_dataset()
    assert X.shape == (1797, 64), "the shapes doesn't match"
    assert y.shape == (1797,), "the shapes doesn't match"

def test_split_dataset():
    x, y = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(x, y)
    assert x_test.shape == (60,64), "the shapes doesn't match"
    assert y_test.shape == (60,), "the shapes doesn't match"
    assert x_train.shape == (1737,64), "the shapes doesn't match"
    assert y_train.shape == (1737,), "the shapes doesn't match"