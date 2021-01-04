from scipy import sparse


def generate_unrolled_sparse_matrix_vector_product(a: sparse.csr_matrix, dtype='double') -> str:
    """Generates the OpenCL code to perform a matrix vector product on the given sparse matrix.
    """

    code = f"""
    __kernel void unrolled_matrix_vector_product(__global {dtype}* x, __global {dtype}* out) {{
        uint gid = get_global_id(0);
        uint gsize = get_global_size(0);
        uint row_gid;
        {dtype} row_val;
    """

    for row in range(a.shape[0]):
        first_index = a.indptr[row]
        last_index = a.indptr[row+1]

        code += f"""
        // Row {row}
        row_gid = {row}*gsize + gid;
        row_val = 0.0;
        """

        for index in range(first_index, last_index):
            col = a.indices[index]
            coef = a.data[index]
            if coef == 1.0:
                rhs = f"x[{col}*gsize+gid]"
            elif coef == -1.0:
                rhs = f"-x[{col}*gsize+gid]"
            else:
                rhs = f"{coef}*x[{col}*gsize+gid]"
            code += f"""
        row_val += {rhs};
            """

        code += """
        out[row_gid] = row_val;
        """

    code += """
    }
    """
    return code
