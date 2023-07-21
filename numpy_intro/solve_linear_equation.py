#!/usr/bin/env python
# coding: utf-8
import numpy as np

eqn = np.array([[7, 5, 3],[3, 2, 5], [1,2,1]], dtype=int)
eqn_val = np.array([120, 70, 20], dtype=int)

x = np.linalg.solve(eqn, eqn_val)
print(x)
det = np.linalg.det(eqn)
print(f"Determinant of matrix A: {det:.2f}")

# eqn_2var = np.array([[1,1], [-6, 2]])
# eqn_2var_value = np.array([4,16])
#
# y = np.linalg.solve(eqn_2var, eqn_2var_value)

A_system = np.hstack((eqn, eqn_val.reshape((3, 1))))

print(eqn_val.reshape((3, 1)))
print(eqn_val)

