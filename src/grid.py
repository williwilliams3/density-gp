import numpy as np
import torch


def make_grid(n, limit):
    x_axis = np.linspace(-limit, limit, n)
    y_axis = np.linspace(-limit, limit, n)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return x_axis, y_axis, xx, yy, points


def nearest_grid_index(points, coord):
    coord = np.asarray(coord)
    return int(np.sum((points - coord) ** 2, axis=1).argmin())


def gradient_norm(values, grid_spacing, n):
    field = values.reshape(n, n)
    gx = torch.zeros_like(field)
    gy = torch.zeros_like(field)

    gx[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * grid_spacing)
    gx[:, 0] = (field[:, 1] - field[:, 0]) / grid_spacing
    gx[:, -1] = (field[:, -1] - field[:, -2]) / grid_spacing

    gy[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2.0 * grid_spacing)
    gy[0, :] = (field[1, :] - field[0, :]) / grid_spacing
    gy[-1, :] = (field[-1, :] - field[-2, :]) / grid_spacing

    return torch.sqrt(gx.pow(2) + gy.pow(2)).reshape(-1)


def image(values, n):
    return values.reshape(n, n)
