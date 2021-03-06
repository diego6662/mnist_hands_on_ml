# -*- coding: utf-8 -*-
"""mnist_chapter4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RgCo-8PPl3r5EjA0ZjTASiPxVyWMdKMx
"""

from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


"""
The objective is train knn model to classified the mnist dataset with
97% as goal
"""

def load_dataset():
  mnist_data = load_digits()
  X = mnist_data.data
  y = mnist_data.target
  return X, y

def split_dataset(X,y):
  x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=60, random_state=42)
  return x_train, x_test, y_train, y_test


def get_best_model(model, x_train, y_train):
  param_grid = {
      'n_neighbors': [*range(1,16)],
      'weights': ['uniform', 'distance']
  }
  grid_clf = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
  grid_clf.fit(x_train, y_train)
  return grid_clf.best_estimator_

def one_shift(x, direction):
  """
  up: u
  right: r
  down: d
  left: l
  """
  if direction == 'u':
    x = x.reshape(8,8)
    x[:-1,:] = x[1:,:]
    x[-1,:] = 0.0
    return x.flatten()
  elif direction == 'r':
    x = x.reshape(8,8)
    x[:,1:] = x[:,:-1]
    x[:,0] = 0.0
    return x.flatten()
  elif direction == 'd':
    x = x.reshape(8,8)
    x[1:,:] = x[:-1,:]
    x[0,:] = 0.0
    return x.flatten()
  else:
    x = x.reshape(8,8)
    x[:,:-1] = x[:,1:]
    x[:,-1] = 0.0
    return x.flatten()

def data_augmentation(x,y):
  directions = ['u', 'r', 'd', 'l']
  x_2 = x.copy()
  for i in directions:
    x_copy = x.copy()
    shifted_data = []
    for j in x_copy:
      shifted_x = one_shift(j,i)
      shifted_data.append(shifted_x)
    shifted_data = np.array(shifted_data)
    x_2 = np.concatenate([x_2,shifted_data])
  y_2 = np.concatenate([y] * 5)
  return x_2, y_2

def train_model(model,x,y):
  best_model = get_best_model(model,x,y)
  best_model.fit(x, y)
  return best_model

def main():
  X, y = load_dataset()
  x_train, x_test, y_train, y_test = split_dataset(X, y)
  x_train, y_train = data_augmentation(x_train, y_train)
  model = KNeighborsClassifier()
  trained_model = train_model(model, x_train, y_train)
  y_hat = trained_model.predict(x_test)
  acc = accuracy_score(y_test, y_hat)
  print(f"Accuracy: {acc:.2f}")

if __name__ == "__main__":
  main()
