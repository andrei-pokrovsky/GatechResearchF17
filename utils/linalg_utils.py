import torch


def pdist2(X: torch.Tensor, Z: torch.Tensor = None) -> torch.Tensor:
    r""" Calculates the pairwise distance between X and Z

    D[b, i, j] = l2 distance X[b, i] and Z[b, j]

    Parameters
    ---------
    X : torch.Tensor
        X is a (B, N, d) tensor.  There are B batches, and N vectors of dimension d
    Z: torch.Tensor
        Z is a (B, M, d) tensor.  If Z is None, then Z = X

    Returns
    -------
    torch.Tensor
        Distance matrix is size (B, N, M)
    """

    if Z is None:
        Z = X
        G = X @ Z.transpose(-2, -1)
        S = X.pow(2).sum(-1, keepdim=True)
        R = S.transpose(-2, -1)
    else:
        G = X @ Z.transpose(-2, -1)
        S = X.pow(2).sum(-1, keepdim=True)
        R = Z.pow(2).sum(-1, keepdim=True).transpose(-2, -1)

    return torch.sqrt(torch.abs(R + S - 2 * G))


def pdist2_slow(X, Z=None):
    if Z is None: Z = X
    D = torch.zeros(X.size(0), X.size(1), Z.size(1))

    for b in range(D.size(0)):
        for i in range(D.size(1)):
            for j in range(D.size(2)):
                D[b, i, j] = torch.dist(X[b, i], Z[b, j])
    return D


if __name__ == "__main__":
    X = torch.randn(2, 5, 3)
    Z = torch.randn(2, 3, 3)

    print(pdist2(X, Z))
    print(pdist2_slow(X, Z))
    print(torch.dist(pdist2(X, Z), pdist2_slow(X, Z)))
