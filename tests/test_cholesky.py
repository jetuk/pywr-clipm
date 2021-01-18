import pyopencl as cl
from scipy import sparse
import numpy as np
from numpy.testing import assert_allclose
from pywr_clipm.sparse import create_sparse_normal_matrix_cholesky_indices, create_sparse_normal_matrix_cholesky_buffers, \
    SparseNormalCholeskyClBuffers, sparse_normal_matrix_cholesky_decomposition
import pytest


def test_create_sparse_normal_matrix_cholesky_buffer(cl_context: cl.Context):
    """Test creating normal equation buffers in the OpenCL context."""

    np.random.seed(12345)
    n = 100
    a = sparse.random(n, n, 0.01, format='csr')

    indices = create_sparse_normal_matrix_cholesky_indices(a)
    cl_buf = create_sparse_normal_matrix_cholesky_buffers(cl_context, indices)
    assert isinstance(cl_buf, SparseNormalCholeskyClBuffers)


@pytest.mark.parametrize('float_type', ['d', 'f'])
def test_cholesky_decomposition(cl_context: cl.Context, cl_queue: cl.CommandQueue, float_type):
    """Test decomposition of normal matrix"""

    np.random.seed(12345)
    n = 10
    gsize = 64
    a = sparse.random(n, n, 0.2, format='csr') + sparse.eye(n)
    wsize = n // 2

    x = np.ones((n, gsize)).astype(float_type)
    z = np.ones((n, gsize)).astype(float_type)
    y = np.ones((n, gsize)).astype(float_type)
    w = np.ones((wsize, gsize)).astype(float_type)

    ldata, lindices, lindptr = sparse_normal_matrix_cholesky_decomposition(
        cl_context, cl_queue, a, x, z, y, w, float_type=float_type
    )

    ad = a.todense()

    for i in range(gsize):
        X = np.asmatrix(np.diag(x[:, i]))
        Z1 = np.asmatrix(np.diag(1 / z[:, i]))

        W = np.r_[w[:, i], np.zeros(n//2)]
        AAt = ad*X*Z1*ad.T
        AAt = np.diag(W/y[:, i]) + AAt
        # Convert expected and result values to sparse matrices for comparison
        expected = sparse.csr_matrix(np.linalg.cholesky(AAt))
        result = sparse.csr_matrix((ldata[:, i], lindices, lindptr))

        np.testing.assert_allclose(result.data, expected.data, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(result.indices, expected.indices)
        np.testing.assert_allclose(result.indptr, expected.indptr)

