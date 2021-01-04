/* Common OpenCL functions for path following interior point method.
 *
 *
 */


#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define matrix(N) __constant uint* N##indptr, __constant uint* N##indices, __constant double* N##data, uint N##size
#define write_matrix(N) __constant uint* N##indptr, __constant uint* N##indices, __global double* N##data

__kernel void matrix_vector_product(
     matrix(A),
    __global double* x,
    __global double* out
) {
    /* Compute out = Ax
     *
     */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    uint row_gid;
    uint row, col;
    uint index, last_index;
    double val;

    for (row=0; row<Asize; row++) {
        row_gid = row*gsize + gid;
        val = 0.0;

        index = Aindptr[row];
        last_index = Aindptr[row+1];

        for (; index < last_index; index++) {
            col = Aindices[index];
            val += Adata[index]*x[col*gsize+gid];
            // val = fma(Adata[index], x[col*gsize+gid], val);
        }

        out[row_gid] = val;
    }
}

double dot_product(__global double* x, __global double* y, int N) {
    /* Return dot product of x and y */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    uint row;
    uint row_gid;
    double val = 0.0;

    for (row=0; row<N; row++) {
        row_gid = row*gsize + gid;
        val += x[row_gid]*y[row_gid];
    }
    return val;
}

void vector_copy(__global double* x, __global double* y, int N) {
    /* Copy vector x in to y */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    uint row;
    uint row_gid;

    for (row=0; row<N; row++) {
        row_gid = row*gsize + gid;
        y[row_gid] = x[row_gid];     
    }
}

void vector_update(__global double* x, __global double* y, double xscale, double yscale, int N) {
    /* x = x*xscale + y*yscale */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    uint row;
    uint row_gid;

    for (row=0; row<N; row++) {
        row_gid = row*gsize + gid;
        // if (gid == 0) {
        //     printf("update %d: %f %f %f %f %f\n", row_gid, xscale, x[row_gid], yscale, y[row_gid], xscale*x[row_gid] + yscale*y[row_gid]);
        // }
        x[row_gid] = xscale*x[row_gid] + yscale*y[row_gid];
    }
}

void vector_set(__global double* x, double scalar, int N) {
    /* x = scalar */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    uint row;
    uint row_gid;

    for (row=0; row<N; row++) {
        row_gid = row*gsize + gid;
        x[row_gid] = scalar;
    }
}

double vector_max(__global double *x, int N) {
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    uint row;
    uint row_gid;

    double val = -INFINITY;

    for (row=0; row<N; row++) {
        row_gid = row*gsize + gid;
        val = max(val, x[row_gid]);
    }
    return val;
}


__kernel void normal_eqn_rhs(
    matrix(A),  // Sparse A matrix
    matrix(AT),  // Sparse transpose of A matrix
    __global double* x,
    __global double* z,
    __global double* y,
    __global double* b,
    __global double* c,
    double mu,
    uint wsize,
    __global double* tmp, // work array size of x
    __global double* out // work array size of b
) {
    /* Compute the right-hand side of the system of primal normal equations

    rhs = -(b - A.dot(x) - mu/y - A.dot(x * (c - At.dot(y) + mu/x)/z))
    */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    uint row, col, index, last_index;
    uint row_gid;
    double val;

    // Calculate tmp = At.dot(y)
    matrix_vector_product(ATindptr, ATindices, ATdata, ATsize, y, tmp);

    // Calculate tmp = x * (c - At.dot(y) + mu/x)/z
    for (row=0; row<ATsize; row++) {
        row_gid = row*gsize + gid;
        tmp[row_gid] = x[row_gid]*(c[row_gid] - tmp[row_gid] + mu/x[row_gid])/z[row_gid];
    }

    // Calculate tmp2 = A.dot(tmp)
    matrix_vector_product(Aindptr, Aindices, Adata, Asize, tmp, out);

    // Compute out = -(b - A.dot(x) - mu/y -out)
    for (row=0; row<Asize; row++) {
        row_gid = row*gsize + gid;

        if (row < wsize) {
            val = mu / y[row_gid];
        } else {
            val = 0.0;
        }

        index = Aindptr[row];
        last_index = Aindptr[row+1];

        while (index < last_index) {
            col = Aindices[index];
            val += Adata[index]*x[col*gsize+gid];
            index += 1;
        }

        out[row_gid] = -(b[row_gid] - val - out[row_gid]);
    }
}

double primal_feasibility(
    matrix(A),  // Sparse A matrix
    __global double* x,
    __global double* w,
    uint wsize,
    __global double* b
) {
    /* Calculate primal-feasibility

        normr = b - A.dot(x)
    */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    uint row, col, index, last_index;
    uint row_gid;
    double val;

    // Compute primal feasibility
    double normr = 0.0;
    for (row=0; row<Asize; row++) {
        row_gid = row*gsize + gid;
        val = b[row_gid];

        if (row < wsize) {
            val -= w[row_gid];
        }

        index = Aindptr[row];
        last_index = Aindptr[row+1];

        while (index < last_index) {
            col = Aindices[index];
            val -= Adata[index]*x[col*gsize+gid];
            index += 1;
        }

        normr += pow(val, 2);
    }

    return sqrt(normr);
}

double dual_feasibility(
    matrix(AT),  // Sparse A matrix
    __global double* y, __global double* c, __global double* z
) {
    /* Calculate dual-feasibility

        norms = c - AT.dot(y) + z
    */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    uint row, col, index, last_index;
    uint row_gid;
    double val;

    // Compute primal feasibility
    double norms = 0.0;
    for (row=0; row<ATsize; row++) {
        row_gid = row*gsize + gid;
        val = c[row_gid] + z[row_gid];

        index = ATindptr[row];
        last_index = ATindptr[row+1];

        while (index < last_index) {
            col = ATindices[index];
            val -= ATdata[index]*y[col*gsize+gid];
            index += 1;
        }

        norms += pow(val, 2);
    }

    return sqrt(norms);
}

double compute_dx_dz_dw(
    uint Asize,
    matrix(AT), // Sparse transpose of A matrix
    __global double* x,
    __global double* z,
    __global double* y,
    __global double* w,
    uint wsize,
    __global double* c,    
    __global double* dy,
    double mu,
    __global double* dx,
    __global double* dz,
    __global double* dw
) {
    /*

        dx = (c - AT.dot(y) - AT.dot(dy) + mu/x)*x/z
        dz = (mu - z*dx)/x - z
        dw = (mu - w*dy)/y - w
    */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    uint row, col, index, last_index;
    uint row_gid;
    double val, val2;

    double theta_xz = 0.0;
    double theta_wy = 0.0;

    for (row=0; row<ATsize; row++) {
        row_gid = row*gsize + gid;
        val = 0.0;
        val2 = 0.0;

        index = ATindptr[row];
        last_index = ATindptr[row+1];

        while (index < last_index) {
            col = ATindices[index];
            val += ATdata[index]*y[col*gsize+gid];
            val2 += ATdata[index]*dy[col*gsize+gid];
            index += 1;
        }

        dx[row_gid] = (c[row_gid] - val - val2 + mu/x[row_gid])*x[row_gid]/z[row_gid];
        dz[row_gid] = (mu - z[row_gid]*dx[row_gid])/x[row_gid] - z[row_gid];

        theta_xz = max(max(theta_xz, -dx[row_gid]/x[row_gid]), -dz[row_gid]/z[row_gid]);  

        //printf("%d x: %g, dx: %g, z: %g, dz: %g, theta: %g", gid, x[row_gid], dx[row_gid], z[row_gid], dz[row_gid], theta_xz);
    }

    for (row=0; row<wsize; row++) {
        row_gid = row*gsize + gid;
        //printf("%d y: %g, dy: %g, theta: %g", gid, y[row_gid], dy[row_gid], theta);        
        dw[row_gid] = (mu - w[row_gid]*dy[row_gid])/y[row_gid] - w[row_gid];
        theta_wy = max(max(theta_wy, -dw[row_gid]/w[row_gid]), -dy[row_gid]/y[row_gid]);
    }

    //printf("%d theta xz: %g, wy: %g", gid, theta_xz, theta_wy);
    for (row=0; row<Asize; row++) {
        row_gid = row*gsize + gid;
        //printf("%d y: %g, dy: %g, -dy/y: %g", gid, y[row_gid], dy[row_gid], -dy[row_gid]/y[row_gid]);
    }

    return max(theta_xz, theta_wy);
}

