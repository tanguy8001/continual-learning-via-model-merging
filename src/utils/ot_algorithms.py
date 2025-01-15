import numpy as np
import ot
import torch

from torch.autograd import Variable


# Adapted from source : https://github.com/gpeyre/SinkhornAutoDiff
def sinkhorn_coupling(cost, epsilon, niter, threshold=1e-3):
    """
    Computes the sinkhorn loss given the cost matrix and assuming uniform distribution.
    :param cost: Cost associated with the linear program
    :param epsilon: regularizer for the sinkhorn
    :param niter: number of iterations
    :param threshold: threshold for stopping the iterations
    :return: Coupling pi the associated cost.
    """
    def M(u, v, C):
        # M_ij = -C_ij + u_i + v_j / epsilon
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / epsilon

    def lse(A):
        # Log sum exp
        return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)

    C = cost
    n = C.size(0)
    m = C.size(1)
    C_norm = C / C.max()

    mu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False)
    nu = Variable(1. / m * torch.FloatTensor(m).fill_(1), requires_grad=False)
    if torch.cuda.is_available():
        mu = mu.cuda()
        nu = nu.cuda()

    u, v, err = 0.0 * mu, 0.* nu, 0.
    actual_nits = 0  # Check the actual iterations required for convergence
    
    for i in range(niter):
        u1 = u
        u = epsilon * (torch.log(mu) - lse(M(u, v, C_norm)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v, C_norm).t()).squeeze()) + v
        err = (u - u1).abs().sum()
        actual_nits += 1
        if err < threshold:
            break

    U, V = u, v
    pi = torch.exp(M(U, V, C_norm))
    cost = torch.sum(pi * C)
            
    return pi, cost


def sinkhorn_coupling_and_potentials(mu, nu, cost, epsilon, niter, threshold=1e-3):
    """
    Computes the sinkhorn loss given the cost matrix and assuming uniform distribution.
    :param mu: First measure
    :param nu: Second measure
    :param cost: Cost associated with the linear program
    :param epsilon: regularizer for the sinkhorn
    :param niter: number of iterations
    :param threshold: threshold for stopping the iterations
    :return: Coupling pi the associated cost, the sinkhorn potentials alpha and beta
    """

    def M(u, v, C):
        # M_ij = -C_ij + u_i + v_j / epsilon
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / epsilon

    def lse(A):
        # Log sum exp
        return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)

    C = cost
    C_norm = C / C.max()

    u, v, err = 0.0 * mu, 0. * nu, 0.
    actual_nits = 0  # Check the actual iterations required for convergence

    for i in range(niter):
        u1 = u
        u = epsilon * (torch.log(mu) - lse(M(u, v, C_norm)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v, C_norm).t()).squeeze()) + v
        err = (u - u1).abs().sum()
        actual_nits += 1
        if err < threshold:
            break

    U, V = u, v
    pi = torch.exp(M(U, V, C_norm))
    cost = torch.sum(pi * C)
    alpha = -u - epsilon/2
    beta = -v - epsilon/2

    return pi, cost, alpha, beta


def ot_emd(mu, nu, cost):
    """
    Computes the sinkhorn loss given the cost matrix.
    :param mu: First measure
    :param nu: Second measure
    :param cost: Cost associated with the linear program
    :returns EMD between the two distributions.
    """
    a = mu.cpu().numpy().astype(np.float64)  # ot.emd needs 64 bits for float
    a /= a.sum()  # Due to numerical issue of using 32 bits
    b = nu.cpu().numpy().astype(np.float64)
    b /= b.sum()
    pi = ot.lp.emd(a=a, b=b, M=cost.cpu().numpy().astype(np.float64))
    pi = torch.from_numpy(pi)
    if mu.is_cuda and torch.cuda.is_available():
        pi = pi.cuda()
    return torch.sum(pi*cost)


def ot_emd_coupling(cost):
    """
    Computes the sinkhorn loss given the cost matrix.
    :param cost: Cost associated with the linear program
    :returns EMD between the two distributions.
    """
    n = cost.size(0)
    m = cost.size(1)
    a = np.ones(n)/n
    b = np.ones(m)/m
    pi = ot.lp.emd(a=a, b=b, M=cost.cpu().numpy().astype(np.float64))
    pi = torch.from_numpy(pi).float()
    if cost.is_cuda and torch.cuda.is_available():
        pi = pi.cuda()
    return pi, torch.sum(pi * cost)


def test():
    n = 100
    m = 200
    mu = torch.rand(n).pow(2)
    mu /= mu.sum()
    nu = torch.rand(m).pow(2)
    nu /= nu.sum()
    cost = torch.rand(n, m).pow(2)
    print(ot_emd(mu, nu, cost))


if __name__ == "__main__":
    test()
