import torch
import numpy as np

def accuracy(predictions, query_labels):
    predictions = torch.argmax(predictions, dim = -1)
    num_data = predictions.size(0)
    true_predict = torch.sum(predictions == query_labels)
    return  true_predict/num_data

def fast_adapt(batch, learner,features, loss, adaptation_steps, shots,
               ways, device=None):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    data =  features(data)
    # Separate data into adaptation/evaluation sets
    support_indices = np.zeros(data.size(0), dtype=bool)
    support_indices[np.arange(shots*ways) * 2] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support_data, support_labels = data[support_indices], labels[support_indices]
    query_data, query_labels = data[query_indices], labels[query_indices]

    for step in range(adaptation_steps):
        train_error = loss(learner(support_data), support_labels)
        learner.adapt(train_error)

    predictions = learner(query_data)
    query_error = loss(predictions, query_labels)
    query_accuracy = accuracy(predictions, query_labels)
    return query_error, query_accuracy

def line(x0, y0, x1, y1):
    "Bresenham's line algorithm"
    points_in_line = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points_in_line.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points_in_line.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points_in_line.append((x, y))
    return points_in_line