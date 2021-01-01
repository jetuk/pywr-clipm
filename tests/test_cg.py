import pyopencl as cl
from scipy import sparse
import numpy as np
from cgpycl.sparse import create_sparse_matrix_buffer, SparseMatrixClBuffer, sparse_matrix_vector_product, \
    SparseNormalMatrixClBuffer, create_sparse_normal_matric_indices, create_sparse_normal_matrix_buffers, \
    normal_matrix_vector_product, normal_conjugate_gradient_solve, normal_equation_rhs


def test_create_sparse_matrix_buffer(cl_context):
    """Test creating a sparse matrix in the OpenCL context."""
    n = 100
    a = sparse.random(n, n, 0.1, format='csr')

    cl_buf = create_sparse_matrix_buffer(cl_context, a)
    assert isinstance(cl_buf, SparseMatrixClBuffer)


def test_sparse_matrix_vector_product(cl_context: cl.Context, cl_queue: cl.CommandQueue):
    """Test sparse matrix-vector product."""
    n = 100
    gsize = 1024
    np.random.seed(1337)
    a = sparse.random(n, n, 0.1, format='csr')
    b = np.random.random((n, gsize))

    c = sparse_matrix_vector_product(cl_context, cl_queue, a, b)

    np.testing.assert_allclose(a.dot(b), c)


def test_create_normal_matrix_buffer(cl_context: cl.Context):
    """Test creating normal equation buffers in the OpenCL context."""

    np.random.seed(12345)
    n = 100
    a = sparse.random(n, n, 0.01, format='csr')

    norm_indices = create_sparse_normal_matric_indices(a)
    cl_buf = create_sparse_normal_matrix_buffers(cl_context, norm_indices)
    assert isinstance(cl_buf, SparseNormalMatrixClBuffer)


def test_normal_matrix_vector_product(cl_context: cl.Context, cl_queue: cl.CommandQueue):
    """Test calcualting normal matrix vector product."""

    np.random.seed(12345)
    n = 100
    gsize = 1024
    a = sparse.random(n, n, 0.01, format='csr')
    wsize = n // 2

    x = np.random.random((n, gsize)).astype(np.float64)
    z = np.random.random((n, gsize)).astype(np.float64)
    b = np.random.random((n, gsize)).astype(np.float64)
    y = np.random.random((n, gsize)).astype(np.float64)
    w = np.random.random((wsize, gsize)).astype(np.float64)

    out = normal_matrix_vector_product(cl_context, cl_queue, a, x, z, y, w, b)

    ad = a.todense()

    for i in range(gsize):
        X = np.asmatrix(np.diag(x[:, i]))
        Z1 = np.asmatrix(np.diag(1 / z[:, i]))

        W = np.r_[w[:, i], np.zeros(n//2)]
        AAt = ad*X*Z1*ad.T
        AAt = np.diag(W/y[:, i]) + AAt
        expected = np.asarray(AAt.dot(b[:, i])).T
        np.testing.assert_allclose(out[:, i], expected[:, 0])


def test_conjugate_gradient_solve(cl_context: cl.Context, cl_queue: cl.CommandQueue):
    """Test solve system of normal equations with conjugate gradient method."""

    np.random.seed(12345)
    n = 10
    gsize = 64
    a = sparse.random(n, n, 0.2, format='csr') + sparse.eye(n)
    wsize = n // 2

    x = np.random.random((n, gsize)).astype(np.float64)
    z = np.random.random((n, gsize)).astype(np.float64)
    y = np.random.random((n, gsize)).astype(np.float64)
    w = np.random.random((wsize, gsize)).astype(np.float64)
    b = np.random.random((n, gsize)).astype(np.float64)

    dy0 = np.zeros((n, gsize)).astype(np.float64)

    dy = normal_conjugate_gradient_solve(cl_context, cl_queue, a, x, z, y, w, b, dy0)

    ad = a.todense()

    for i in range(gsize):
        X = np.asmatrix(np.diag(x[:, i]))
        Z1 = np.asmatrix(np.diag(1 / z[:, i]))

        W = np.r_[w[:, i], np.zeros(n//2)]
        AAt = ad*X*Z1*ad.T
        AAt = np.diag(W/y[:, i]) + AAt
        expected = np.linalg.solve(np.asarray(AAt), b[:, i])
        np.testing.assert_allclose(dy[:, i], expected, atol=1e-6, rtol=1e-6)


def test_normal_equation_rhs(cl_context: cl.Context, cl_queue: cl.CommandQueue):

    np.random.seed(12345)
    n = 10
    gsize = 1024
    a = sparse.random(n, n, 0.2, format='csr') + sparse.eye(n)

    ad = a.todense()
    adt = ad.T

    x = np.ones((n, gsize)).astype(np.float64)
    z = np.ones_like(x)

    b = np.random.random((n, gsize)).astype(np.float64)
    c = np.random.random((n, gsize)).astype(np.float64)

    y = np.ones(b.shape)  # * 0.0001

    delta = 0.1

    rhs = normal_equation_rhs(cl_context, cl_queue, a, x, z, y, b, c, delta, n//2)

    for i in range(gsize):
        # Create system of primal normal equations
        gamma = np.dot(z[:, i], x[:, i])
        mu = delta * gamma / c.shape[0]
        muy = np.r_[mu / y[:n//2, i], np.zeros(n//2)]

        expected = -(b[:, i] - a.dot(x[:, i]) - muy -
                     a.dot(x[:, i] * (c[:, i] - np.array(adt.dot(y[:, i]))[0, :] + mu / x[:, i]) / z[:, i]))

        np.testing.assert_allclose(rhs[:, i], expected, atol=1e-6, rtol=1e-6)
