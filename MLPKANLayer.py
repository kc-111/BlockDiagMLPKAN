import torch
import torch.nn as nn

# Making this use sparse module would help a lot.
class MLPKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, gain=1.0, activation=nn.SiLU()):
        super(MLPKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.M1, self.mask1, self.M2, self.mask2, self.b1, self.b2 = get_sparse_MLP_matrix(input_dim, output_dim, degree + 1, gain=gain)
        self.M1 = nn.Parameter(self.M1)
        self.M2 = nn.Parameter(self.M2)
        self.b1 = nn.Parameter(self.b1)
        self.b2 = nn.Parameter(self.b2)
        self.mask1.requires_grad = False
        self.mask2.requires_grad = False
        self.activation = activation

    def forward(self, x): # (inputdim, batch_size)
        cur_device = self.M1.device
        y = torch.mm(self.M1*self.mask1.to(cur_device), x) + self.b1 # (inputdim*outdim*N, batch_size)
        # y = torch.sparse.mm(self.M1, x) + self.b1.to_sparse() # (inputdim*outdim*N, batch_size)
        y = self.activation(y)
        y = torch.mm(self.M2*self.mask2.to(cur_device), y) + self.b2 # (outdim, batch_size)
        # y = torch.sparse.mm(self.M2, y) + self.b2.to_sparse() # (outdim, batch_size)
        return y

def get_sparse_MLP_matrix(in_dim, out_dim, N, gain=1.0):
    # Input layer
    matrix = torch.block_diag(*[torch.randn(N, 1) for _ in range(in_dim)]).repeat(out_dim, 1)
    matrix_nonzeros = torch.nonzero(matrix, as_tuple=True)
    nn.init.xavier_uniform_(matrix, gain=gain) # Xavier uniform
    # Create a mask tensor of the same shape as matrix
    mask1 = torch.zeros_like(matrix, dtype=torch.bool)
    mask1[matrix_nonzeros] = True
    matrix[~mask1] = 0.0
    # matrix = torch.sparse_coo_tensor(torch.nonzero(matrix).t(), matrix2[matrix != 0], matrix.shape)
    # matrix = matrix.coalesce()
    # Get biases
    biases1 = torch.randn(in_dim*out_dim*N, 1)

    # Output layer
    matrix2 = torch.block_diag(*[torch.randn(1, in_dim*N) for _ in range(out_dim)])
    matrix_nonzeros2 = torch.nonzero(matrix2, as_tuple=True)
    nn.init.xavier_uniform_(matrix2, gain=gain) # Xavier uniform
    # Create a mask tensor of the same shape as matrix
    mask2 = torch.zeros_like(matrix2, dtype=torch.bool)
    mask2[matrix_nonzeros2] = True
    matrix2[~mask2] = 0.0
    # matrix2 = torch.sparse_coo_tensor(torch.nonzero(matrix2).t(), matrix2[matrix2 != 0], matrix2.shape)
    # matrix2 = matrix2.coalesce()
    # Get biases
    biases2 = torch.zeros(out_dim, 1)
    return matrix, mask1, matrix2, mask2, biases1, biases2
    