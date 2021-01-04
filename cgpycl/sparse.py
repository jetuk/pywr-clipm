import pyopencl as cl
from scipy import sparse
import numpy as np
from dataclasses import dataclass
import time
from typing import Tuple
from .cl import get_cl_program
from . import sparse_gen

MF = cl.mem_flags


@dataclass
class SparseMatrixClBuffer:
    data: cl.Buffer
    indices: cl.Buffer
    indptr: cl.Buffer
    nrows: np.uint32


def create_sparse_matrix_buffer(cl_context: cl.Context, a: sparse.csr_matrix) -> SparseMatrixClBuffer:
    """Create and populate buffers in the OpenCL context for the given sparse matrix"""

    data = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=a.data)
    indices = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=a.indices.astype(np.uint32))
    indptr = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=a.indptr.astype(np.uint32))
    return SparseMatrixClBuffer(data, indices, indptr, np.uint32(a.shape[0]))


@dataclass
class SparseNormalMatrixClBuffer:
    rowptr: cl.Buffer
    diagptr: cl.Buffer
    colptr: cl.Buffer
    colindices: cl.Buffer
    indices: cl.Buffer
    indptr1: cl.Buffer
    indptr2: cl.Buffer


@dataclass
class SparseNormalMatrix:
    rowptr: np.ndarray
    diagptr: np.ndarray
    colptr: np.ndarray
    colindices: np.ndarray
    indices: np.ndarray
    indptr1: np.ndarray
    indptr2: np.ndarray


def create_sparse_normal_matrix_indices(a: sparse.csr_matrix) -> SparseNormalMatrix:
    """Create and populate buffers in the OpenCL context for the normal matrix of the given sparse matrix."""

    row_indptrptr = [0]
    diag_indptrptr = []
    col_indptrptr = [0]
    col_indices = []
    indices = []
    indptr_i = []
    indptr_j = []

    for i in range(a.shape[0]):
        for j in range(a.shape[0]):

            ii = a.indptr[i]
            ii_max = a.indptr[i + 1]
            jj = a.indptr[j]
            jj_max = a.indptr[j + 1]
            non_zero = False

            while (ii < ii_max) and (jj < jj_max):
                ik = a.indices[ii]
                jk = a.indices[jj]

                if ik == jk:
                    indptr_i.append(ii)
                    indptr_j.append(jj)
                    indices.append(ik)
                    non_zero = True
                    ii += 1
                    jj += 1
                elif ik < jk:
                    ii += 1
                else:
                    jj += 1
            if non_zero:
                col_indptrptr.append(len(indices))
                col_indices.append(j)
            if i == j:
                diag_indptrptr.append(len(col_indptrptr) - 2)
        row_indptrptr.append(len(col_indptrptr) - 1)

    assert len(indices) < 65536
    print(f'row_indptrptr: {len(row_indptrptr)}, col_indptrptr: {len(col_indptrptr)}, indices: {len(indices)},')
    # Create the CL buffers
    return SparseNormalMatrix(
        np.array(row_indptrptr, dtype=np.uint32),
        np.array(diag_indptrptr, dtype=np.uint32),
        np.array(col_indptrptr, dtype=np.uint32),
        np.array(col_indices, dtype=np.uint32),
        np.array(indices, dtype=np.uint32),
        np.array(indptr_i, dtype=np.uint32),
        np.array(indptr_j, dtype=np.uint32),
    )


def create_sparse_normal_matrix_buffers(cl_context: cl.Context, indices: SparseNormalMatrix
                                        ) -> SparseNormalMatrixClBuffer:

    row_indptrptr = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.rowptr)
    diag_indptrptr = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.diagptr)
    col_indptrptr = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.colptr)
    col_indices = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.colindices)
    ind = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.indices)
    indptr_i = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.indptr1)
    indptr_j = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.indptr2)

    return SparseNormalMatrixClBuffer(
        row_indptrptr,
        diag_indptrptr,
        col_indptrptr,
        col_indices,
        ind,
        indptr_i,
        indptr_j
    )


@dataclass
class SparseNormalCholeskyIndices:
    anorm_indptr: np.ndarray
    anorm_indptr_i: np.ndarray
    anorm_indptr_j: np.ndarray
    anorm_indices: np.ndarray
    ldecomp_indptr: np.ndarray
    ldecomp_indptr_i: np.ndarray
    ldecomp_indptr_j: np.ndarray
    lindptr: np.ndarray
    ldiag_indptr: np.ndarray
    lindices: np.ndarray
    ltindptr: np.ndarray
    ltindices: np.ndarray
    ltmap: np.ndarray


def create_sparse_normal_matrix_cholesky_indices(a: sparse.csr_matrix) -> SparseNormalCholeskyIndices:
    """Compute the calculation indices for the LDL decomposition of A*AT"""

    a_indptr = [0]
    a_indptr_i = []
    a_indptr_j = []
    a_indices = []
    l_indptr = [0]
    l_indptr_i = []
    l_indptr_j = []
    # Entries of the L matrix
    indptr = [0]
    diag_indptr = []
    indices = []

    for i in range(a.shape[0]):
        for j in range(i+1):
            # Search for matching indices in the a matrix for element AAT[i, j]
            ii = a.indptr[i]
            ii_max = a.indptr[i + 1]
            jj = a.indptr[j]
            jj_max = a.indptr[j + 1]
            non_zero = False

            while (ii < ii_max) and (jj < jj_max):
                ik = a.indices[ii]
                jk = a.indices[jj]

                if ik == jk:
                    a_indptr_i.append(ii)
                    a_indptr_j.append(jj)
                    a_indices.append(ik)
                    non_zero = True
                    ii += 1
                    jj += 1
                elif ik < jk:
                    ii += 1
                else:
                    jj += 1

            # Now search for matching indices for the L[i, k]*L[j, k]
            ii = indptr[i]
            ii_max = len(indices)
            jj = indptr[j]
            if i == j:
                jj_max = ii_max
            else:
                jj_max = indptr[j + 1]

            while (ii < ii_max) and (jj < jj_max):
                ik = indices[ii]
                jk = indices[jj]

                if ik == jk:
                    l_indptr_i.append(ii)
                    l_indptr_j.append(jj)
                    non_zero = True
                    ii += 1
                    jj += 1
                elif ik < jk:
                    ii += 1
                else:
                    jj += 1

            if non_zero:
                a_indptr.append(len(a_indptr_i))
                l_indptr.append(len(l_indptr_i))
                indices.append(j)
            if i == j:
                diag_indptr.append(len(indices) - 1)

        indptr.append(len(indices))

    dtype = np.uint32
    # Compute transpose mapping
    l = sparse.csr_matrix((np.ones(len(indices)), indices, indptr))
    lt = l.transpose().tocsr()
    ltmap = np.argsort(l.indices, kind='mergesort').astype(dtype)

    return SparseNormalCholeskyIndices(
        np.array(a_indptr, dtype=dtype),
        np.array(a_indptr_i, dtype=dtype),
        np.array(a_indptr_j, dtype=dtype),
        np.array(a_indices, dtype=dtype),
        np.array(l_indptr, dtype=dtype),
        np.array(l_indptr_i, dtype=dtype),
        np.array(l_indptr_j, dtype=dtype),
        np.array(indptr, dtype=dtype),
        np.array(diag_indptr, dtype=dtype),
        np.array(indices, dtype=dtype),
        lt.indptr.astype(dtype),
        lt.indices.astype(dtype),
        ltmap
    )


@dataclass
class SparseNormalCholeskyClBuffers:
    anorm_indptr: cl.Buffer
    anorm_indptr_i: cl.Buffer
    anorm_indptr_j: cl.Buffer
    anorm_indices: cl.Buffer
    ldecomp_indptr: cl.Buffer
    ldecomp_indptr_i: cl.Buffer
    ldecomp_indptr_j: cl.Buffer
    lindptr: cl.Buffer
    ldiag_indptr: cl.Buffer
    lindices: cl.Buffer
    ltindptr:  cl.Buffer
    ltindices:  cl.Buffer
    ltmap:  cl.Buffer


def create_sparse_normal_matrix_cholesky_buffers(cl_context: cl.Context,
                                                 indices: SparseNormalCholeskyIndices) -> SparseNormalCholeskyClBuffers:

    return SparseNormalCholeskyClBuffers(
        cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.anorm_indptr),
        cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.anorm_indptr_i),
        cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.anorm_indptr_j),
        cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.anorm_indices),
        cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.ldecomp_indptr),
        cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.ldecomp_indptr_i),
        cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.ldecomp_indptr_j),
        cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.lindptr),
        cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.ldiag_indptr),
        cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.lindices),
        cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.ltindptr),
        cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.ltindices),
        cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=indices.ltmap),
    )


def sparse_normal_matrix_cholesky_decomposition(
        cl_context: cl.Context, cl_queue: cl.CommandQueue, a: sparse.csr_matrix,
        x: np.ndarray, z: np.ndarray, y: np.ndarray, w: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    a_buf = create_sparse_matrix_buffer(cl_context, a)
    indices = create_sparse_normal_matrix_cholesky_indices(a)
    decomp_buf = create_sparse_normal_matrix_cholesky_buffers(cl_context, indices)

    x_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=x)
    z_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=z)
    y_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=y)
    w_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=w)

    gsize = x.shape[1]

    # Create a local vector and context buffer for the result
    ldata = np.zeros((indices.lindices.shape[0], gsize))
    ldata_buf = cl.Buffer(cl_context, MF.WRITE_ONLY, ldata.nbytes)

    # Get the cl program
    program = get_cl_program(cl_context, filename='path_following_direct.cl')
    t0 = time.perf_counter()
    evt = program.normal_matrix_cholesky_decomposition(
        cl_queue,
        (gsize,),
        None,
        a_buf.data,
        a_buf.nrows,
        decomp_buf.anorm_indptr,
        decomp_buf.anorm_indptr_i,
        decomp_buf.anorm_indptr_j,
        decomp_buf.anorm_indices,
        decomp_buf.ldecomp_indptr,
        decomp_buf.ldecomp_indptr_i,
        decomp_buf.ldecomp_indptr_j,
        x_buf,
        z_buf,
        y_buf,
        w_buf,
        np.uint32(w.shape[0]),
        decomp_buf.lindptr,
        decomp_buf.ldiag_indptr,
        decomp_buf.lindices,
        ldata_buf
    )
    evt.wait()
    print(f'matrix-vector product took: {time.perf_counter() - t0}s')
    cl.enqueue_copy(cl_queue, ldata, ldata_buf)

    return ldata, indices.lindices, indices.lindptr


def sparse_matrix_vector_product(cl_context: cl.Context, cl_queue: cl.CommandQueue,
                                 a: sparse.csr_matrix, b: np.ndarray) -> np.ndarray:
    """Compute the matrix-vector product of A and b.
    """
    # Copy the sparse matrix and vector to the context
    a_buf = create_sparse_matrix_buffer(cl_context, a)
    b_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=b)

    gsize = b.shape[1]

    # Create a local vector and context buffer for the result
    c = np.zeros((a.shape[0], gsize))
    c_buf = cl.Buffer(cl_context, MF.WRITE_ONLY, c.nbytes)

    # Get the cl program
    program = get_cl_program(cl_context)
    t0 = time.perf_counter()
    evt = program.matrix_vector_product(cl_queue, (gsize,), None, a_buf.indptr, a_buf.indices, a_buf.data, a_buf.nrows,
                                        b_buf, c_buf)
    evt.wait()
    print(f'matrix-vector product took: {time.perf_counter() - t0}s')
    cl.enqueue_copy(cl_queue, c, c_buf)
    return c


def unrolled_sparse_matrix_vector_product(cl_context: cl.Context, cl_queue: cl.CommandQueue,
                                          a: sparse.csr_matrix, b: np.ndarray) -> np.ndarray:
    """Compute the matrix-vector product of A and b using an unrolled kernel.
    """
    # Only b must be copied to the device
    b_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=b)

    gsize = b.shape[1]

    # Create a local vector and context buffer for the result
    c = np.zeros((a.shape[0], gsize))
    c_buf = cl.Buffer(cl_context, MF.WRITE_ONLY, c.nbytes)

    code = sparse_gen.generate_unrolled_sparse_matrix_vector_product(a)
    # Compiled the cl program
    program = get_cl_program(cl_context, code=code)
    t0 = time.perf_counter()
    evt = program.unrolled_matrix_vector_product(cl_queue, (gsize,), None, b_buf, c_buf)
    evt.wait()
    print(f'unrolled matrix-vector product took: {time.perf_counter() - t0}s')
    cl.enqueue_copy(cl_queue, c, c_buf)
    return c


def normal_matrix_vector_product(
        cl_context: cl.Context, cl_queue: cl.CommandQueue, a: sparse.csr_matrix,
        x: np.ndarray, z: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Compute the matrix-vector product of the c = (A(x/z)A^T)b"""

    # Copy the sparse matrix, it's normal indices  and vector to the context
    a_buf = create_sparse_matrix_buffer(cl_context, a)
    norm_indices = create_sparse_normal_matrix_indices(a)
    norm_a_buf = create_sparse_normal_matrix_buffers(cl_context, norm_indices)
    x_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=x)
    z_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=z)
    y_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=y)
    w_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=w)
    b_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=b)

    gsize = b.shape[1]

    # Create a local vector and context buffer for the result
    c = np.zeros((a.shape[0], gsize))
    c_buf = cl.Buffer(cl_context, MF.WRITE_ONLY, c.nbytes)

    # Get the cl program
    program = get_cl_program(cl_context)
    gsize = b.shape[1]
    t0 = time.perf_counter()
    evt = program.normal_matrix_vector_product(cl_queue, (gsize,), None, a_buf.data, a_buf.nrows, norm_a_buf.rowptr,
                                               norm_a_buf.colptr,
                                               norm_a_buf.colindices, norm_a_buf.indices, norm_a_buf.indptr1,
                                               norm_a_buf.indptr2,
                                               x_buf, z_buf, y_buf, w_buf, np.uint32(w.shape[0]), b_buf, c_buf)
    evt.wait()
    print(f'vecotr-matrix-vector product took: {time.perf_counter() - t0}s')
    cl.enqueue_copy(cl_queue, c, c_buf)
    return c


def normal_conjugate_gradient_solve(
        cl_context: cl.Context, cl_queue: cl.CommandQueue, a: sparse.csr_matrix,
        x: np.ndarray, z: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray, dy0: np.ndarray
) -> np.ndarray:
    """Solve system of normal equations using conjugate gradient method."""

    # Copy the sparse matrix, it's normal indices  and vector to the context
    a_buf = create_sparse_matrix_buffer(cl_context, a)
    norm_indices = create_sparse_normal_matrix_indices(a)
    norm_a_buf = create_sparse_normal_matrix_buffers(cl_context, norm_indices)
    x_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=x)
    z_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=z)
    y_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=y)
    w_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=w)
    b_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=b)

    # Create a local vector and context buffer for the result
    dy_buf = cl.Buffer(cl_context, MF.READ_WRITE | MF.COPY_HOST_PTR, hostbuf=dy0)

    # Work arrays
    r_buf = cl.Buffer(cl_context, MF.WRITE_ONLY, dy0.nbytes)
    p_buf = cl.Buffer(cl_context, MF.WRITE_ONLY, dy0.nbytes)
    s_buf = cl.Buffer(cl_context, MF.WRITE_ONLY, dy0.nbytes)

    # Get the cl program
    program = get_cl_program(cl_context)
    gsize = b.shape[1]
    t0 = time.perf_counter()
    evt = program.normal_eqn_conjugate_gradient(cl_queue, (gsize,), None,
                                                a_buf.data, a_buf.nrows, norm_a_buf.rowptr, norm_a_buf.diagptr,
                                                norm_a_buf.colptr, norm_a_buf.colindices, norm_a_buf.indices,
                                                norm_a_buf.indptr1, norm_a_buf.indptr2, x_buf, z_buf, y_buf, w_buf,
                                                np.uint32(w.shape[0]), b_buf, dy_buf, r_buf, p_buf, s_buf)
    evt.wait()
    print(f'vecotr-matrix-vector product took: {time.perf_counter() - t0}s')
    dy = np.zeros_like(dy0)
    cl.enqueue_copy(cl_queue, dy, dy_buf)
    return dy


def normal_equation_rhs(
        cl_context: cl.Context, cl_queue: cl.CommandQueue, a: sparse.csr_matrix,
        x: np.ndarray, z: np.ndarray, y: np.ndarray, b: np.ndarray, c: np.ndarray,
        delta: float, wsize: int,
) -> np.ndarray:
    # Copy the sparse matrix, it's normal indices  and vector to the context
    a_buf = create_sparse_matrix_buffer(cl_context, a)
    at_buf = create_sparse_matrix_buffer(cl_context, a.T.tocsr())

    x_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=x)
    z_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=z)
    y_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=y)

    b_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=b)
    c_buf = cl.Buffer(cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=c)

    tmp_buf = cl.Buffer(cl_context, MF.WRITE_ONLY, x.nbytes)
    rhs_buf = cl.Buffer(cl_context, MF.WRITE_ONLY, b.nbytes)

    # Get the cl program
    program = get_cl_program(cl_context)
    gsize = b.shape[1]
    program.normal_eqn_rhs(cl_queue, (gsize,), None,
                           a_buf.indptr, a_buf.indices, a_buf.data, a_buf.nrows,
                           at_buf.indptr, at_buf.indices, at_buf.data, at_buf.nrows,
                           x_buf, z_buf, y_buf, b_buf, c_buf, np.float64(delta), np.uint32(wsize), tmp_buf, rhs_buf)

    rhs = np.zeros_like(b)
    cl.enqueue_copy(cl_queue, rhs, rhs_buf)
    return rhs
