import torch

from betty.utils import to_vec


def darts(vector, curr, prev):
    """
    Approximate the matrix-vector multiplication with the best response Jacobian by the
    finite difference method. More specifically, we modified the finite difference method proposed
    in `DARTS: Differentiable Architecture Search <https://arxiv.org/pdf/1806.09055.pdf>`_ by
    re-interpreting it from the implicit differentiation perspective. Empirically, this method
    achieves better memory efficiency, training wall time, and test accuracy that other methods.

    :param vector:
        Vector with which matrix-vector multiplication with best-response Jacobian (matrix) would
        be performed.
    :type vector: Sequence of Tensor
    :param curr: A current level problem
    :type curr: Problem
    :param prev: A directly lower-level problem to the current problem
    :type prev: Problem
    :return: (Intermediate) gradient
    :rtype: Sequence of Tensor
    """
    config = curr.config
    R = config.darts_alpha
    eps = R / to_vec(list(filter(lambda elt: elt is not None, vector))).norm()

    # positive
    for p, v in zip(curr.trainable_parameters(), vector):
        if v is not None:
            p.data.add_(v.data, alpha=eps)
    loss_p = curr.training_step_exec(curr.cur_batch)
    grad_p = torch.autograd.grad(loss_p, prev.trainable_parameters(), allow_unused=True)

    # negative
    for p, v in zip(curr.trainable_parameters(), vector):
        if v is not None:
            p.data.add_(v.data, alpha=-2 * eps)
    loss_n = curr.training_step_exec(curr.cur_batch)
    grad_n = torch.autograd.grad(loss_n, prev.trainable_parameters(), allow_unused=True)

    # reverse weight change
    for p, v in zip(curr.trainable_parameters(), vector):
        if v is not None:
            p.data.add(v.data, alpha=eps)

    def f(x,y):
        if x is None and y is None:
            return None
        if x is None:
            x = 0
        if y is None:
            y = 0
        return (x - y).div_(2 * eps)
    
    implicit_grad = [f(x,y) for x, y in zip(grad_n, grad_p)]

    return implicit_grad
