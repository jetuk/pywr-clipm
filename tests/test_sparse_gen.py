from scipy import sparse
import numpy as np
from pywr_clipm.sparse_gen import generate_unrolled_sparse_matrix_vector_product
from pywr_clipm.sparse import unrolled_sparse_matrix_vector_product
from pywr_clipm.cl import get_cl_program


def test_gen_matrix_vector_product(cl_context, cl_queue):
    n = 100
    np.random.seed(1337)
    a = sparse.random(n, n, 0.1, format='csr')

    code = generate_unrolled_sparse_matrix_vector_product(a)
    # Ensure the generated code compiles
    get_cl_program(cl_context, code=code)


def test_unrolled_matrix_vector_product(cl_context, cl_queue):
    n = 100
    gsize = 1024
    np.random.seed(1337)
    a = sparse.random(n, n, 0.1, format='csr')
    b = np.random.random((n, gsize))

    c = unrolled_sparse_matrix_vector_product(cl_context, cl_queue, a, b)

    np.testing.assert_allclose(a.dot(b), c)


def test_unrolled_binary_matrix_vector_product(cl_context, cl_queue):
    n = 100
    gsize = 1024
    np.random.seed(1337)
    a = sparse.random(n, n, 0.1, format='csr')
    a.data = np.ones_like(a.data)
    b = np.random.random((n, gsize))

    c = unrolled_sparse_matrix_vector_product(cl_context, cl_queue, a, b)

    np.testing.assert_allclose(a.dot(b), c)
