import torch
from scipy.sparse import csr_matrix

def hybrid_spmv(values, row_ptr, col_indices, x, threshold=None):
    num_rows = len(row_ptr) - 1
    nnz = len(values)
    
    # Convert CSR to dense for simplicity (in practice, avoid this)
    csr_mat = csr_matrix((values, col_indices, row_ptr), shape=(num_rows, x.shape[0]))
    nnz_per_row = csr_mat.getnnz(axis=1)
    
    # Determine threshold dynamically if not provided
    if threshold is None:
        threshold = int(max(nnz_per_row) / 2)
    
    # Split into ELL and COO
    ell_rows = nnz_per_row <= threshold
    coo_rows = ~ell_rows
    
    # ELL part (uniform nnz per row)
    max_nnz_ell = nnz_per_row[ell_rows].max() if ell_rows.any() else 0
    ell_indices = []
    ell_values = []
    for i in range(num_rows):
        if ell_rows[i]:
            row_cols = col_indices[row_ptr[i]:row_ptr[i+1]]
            row_vals = values[row_ptr[i]:row_ptr[i+1]]
            ell_indices.extend(row_cols.tolist() + [-1] * (max_nnz_ell - len(row_cols)))
            ell_values.extend(row_vals.tolist() + [0] * (max_nnz_ell - len(row_vals)))
    
    # COO part (irregular rows)
    coo_row_indices = []
    coo_col_indices = []
    coo_values = []
    for i in range(num_rows):
        if coo_rows[i]:
            coo_row_indices.extend([i] * (row_ptr[i+1] - row_ptr[i]))
            coo_col_indices.extend(col_indices[row_ptr[i]:row_ptr[i+1]].tolist())
            coo_values.extend(values[row_ptr[i]:row_ptr[i+1]].tolist())
    
    # Convert to tensors
    device = x.device
    if ell_values:
        ell_values = torch.tensor(ell_values, device=device).reshape(-1, max_nnz_ell)
        ell_indices = torch.tensor(ell_indices, device=device).reshape(-1, max_nnz_ell)
        ell_result = torch.zeros(ell_rows.sum(), device=device)
        for j in range(max_nnz_ell):
            valid = ell_indices[:, j] >= 0
            ell_result[valid] += ell_values[valid, j] * x[ell_indices[valid, j]]
    else:
        ell_result = torch.tensor([], device=device)
    
    if coo_values:
        coo_indices = torch.tensor([coo_row_indices, coo_col_indices], device=device)
        coo_values = torch.tensor(coo_values, device=device)
        coo_mat = torch.sparse_coo_tensor(coo_indices, coo_values, (num_rows, x.shape[0]))
        coo_result = torch.sparse.mm(coo_mat, x.unsqueeze(1)).squeeze(1)
    else:
        coo_result = torch.zeros(num_rows, device=device)
    
    # Combine results
    y = torch.zeros(num_rows, device=device)
    if ell_values:
        y[ell_rows] = ell_result
    if coo_values:
        y[coo_rows] = coo_result
    
    return y

# Example usage
values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device='cuda')
row_ptr = torch.tensor([0, 2, 3, 5, 6], device='cuda')
col_indices = torch.tensor([0, 1, 2, 1, 3, 2], device='cuda')
x = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')

y = hybrid_spmv(values, row_ptr, col_indices, x)
print(y)