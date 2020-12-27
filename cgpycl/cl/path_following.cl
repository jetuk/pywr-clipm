#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define _c __constant
#define _g __global
#define matrix(N) __constant ushort* N##indptr, __constant ushort* N##indices, __constant double* N##data, ushort N##size
#define write_matrix(N) __constant ushort* N##indptr, __constant ushort* N##indices, __global double* N##data



__kernel void matrix_vector_product(
     matrix(A),
    __global double* x,
    __global double* out
) {
    /* Compute y = Ax
     *
     */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    uint row_gid;
    ushort row, col;
    ushort index, last_index;
    double val;

    for (row=0; row<Asize; row++) {
        row_gid = row*gsize + gid;
        val = 0.0;

        index = Aindptr[row];
        last_index = Aindptr[row+1];

        while (index < last_index) {
            col = Aindices[index];
            val += Adata[index]*x[col*gsize+gid];
            index += 1;
        }

        out[row_gid] = val;
    }
}


__kernel void normal_matrix_vector_product(
    __constant double* Adata, ushort Asize,
    __constant ushort* Anorm_rowptr,
    __constant ushort* Anorm_colptr,
    __constant ushort* Anorm_colindices,
    __constant ushort* Anorm_indices,
    __constant ushort* Anorm_indptr_i,
    __constant ushort* Anorm_indptr_j,
    __global double* x,
    __global double* z,
    __global double* b,
    __global double* out
) {
    /* Compute the product of the normal equations (AA^T) with vector b

    i.e. (A(x/z)A^T)b
    */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    ushort row, col, ii, jj, ik;
    uint row_gid, kk;
    ushort col_start, col_end, col_ptr, col_ptr_end;
    double val, inner_val;

    for (row=0; row<Asize; row++) {
        row_gid = row*gsize + gid;
        val = 0.0;
        
        col_ptr = Anorm_rowptr[row];
        col_ptr_end = Anorm_rowptr[row + 1];

        //printf("row [%d]; col_ptr, col_ptr_end [%d, %d]\n", row, col_ptr, col_ptr_end);

        while (col_ptr < col_ptr_end) {
            col_start = Anorm_colptr[col_ptr];
            col_end = Anorm_colptr[col_ptr + 1];
            col = Anorm_colindices[col_ptr];
            inner_val = 0.0;

            while (col_start < col_end) {
                ii = Anorm_indptr_i[col_start];
                jj = Anorm_indptr_j[col_start];
                ik = Anorm_indices[col_start];   
                kk = gsize*ik + gid;             
                inner_val += Adata[ii]*Adata[jj]*x[kk]/z[kk];
                col_start += 1;
            }

            //printf("row [%d]; col_start, col_end [%d, %d]\n", row, col_start, col_end);
            val += inner_val*b[col*gsize + gid];
            col_ptr += 1;
            
        }

        out[row_gid] = val;
        //break;
    }
}


double vector_normal_eqn_vector_product(
    __constant double* Adata,
    ushort Asize,
    __constant ushort* Anorm_rowptr,
    __constant ushort* Anorm_colptr,
    __constant ushort* Anorm_colindices,
    __constant ushort* Anorm_indices,
    __constant ushort* Anorm_indptr_i,
    __constant ushort* Anorm_indptr_j,
    __global double* x,
    __global double* z,
    __global double* b
) {
    /* Compute the product of the normal equations (AA^T) with vector b

    i.e. b(A(x/z)A^T)b
    */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    ushort row, col, ii, jj, ik;
    uint row_gid, kk;
    ushort col_start, col_end, col_ptr, col_ptr_end;
    double val = 0.0;
    double inner_val;

    for (row=0; row<Asize; row++) {
        row_gid = row*gsize + gid;
        
        col_ptr = Anorm_rowptr[row];
        col_ptr_end = Anorm_rowptr[row + 1];

        while (col_ptr < col_ptr_end) {
            col_start = Anorm_colptr[col_ptr];
            col_end = Anorm_colptr[col_ptr + 1];
            col = Anorm_colindices[col_ptr];
            inner_val = 0.0;

            while (col_start < col_end) {
                ii = Anorm_indptr_i[col_start];
                jj = Anorm_indptr_j[col_start];
                ik = Anorm_indices[col_start];   
                kk = gsize*ik + gid;             
                inner_val += Adata[ii]*Adata[jj]*x[kk]/z[kk];
                col_start += 1;
            }

            //printf("row [%d]; col_start, col_end [%d, %d]\n", row, col_start, col_end);
            val += b[row_gid]*inner_val*b[col*gsize + gid];
            col_ptr += 1;
            
        }        

    }

    return val;
}


void residuals(
    __constant double* Adata,
    ushort Asize,
    __constant ushort* Anorm_rowptr,
    __constant ushort* Anorm_colptr,
    __constant ushort* Anorm_colindices,
    __constant ushort* Anorm_indices,
    __constant ushort* Anorm_indptr_i,
    __constant ushort* Anorm_indptr_j,
    __global double* x,
    __global double* z,
    __global double* b,
    __global double* y,
    __global double* out
) {
    /* Compute the residual, r = b - (A(x/z)A^T)y
    */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    uint row_gid;
    ushort row;

    // Compute the matrix-vector product (A(x/z)A^T)y
    normal_matrix_vector_product(Adata, Asize, Anorm_rowptr, Anorm_colptr, Anorm_colindices, Anorm_indices,
                                 Anorm_indptr_i, Anorm_indptr_j, x, z, y, out);
    
    for (row=0; row<Asize; row++) {
        row_gid = row*gsize + gid;

        //printf("residual %d: %f %f %f\n", row_gid, b[row_gid], out[row_gid], b[row_gid] - out[row_gid]);
        out[row_gid] = b[row_gid] - out[row_gid];

    }
}


void preconditioned_residuals(
    __constant double* Adata,
    ushort Asize,
    __constant ushort* Anorm_diagptr,
    __constant ushort* Anorm_colptr,
    __constant ushort* Anorm_colindices,
    __constant ushort* Anorm_indices,
    __constant ushort* Anorm_indptr_i,
    __constant ushort* Anorm_indptr_j,
    __global double* x,
    __global double* z,
    __global double* r,
    __global double* out
) {
    /* Compute z = M^{-1}r

    Where M = diag(A)
    */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    ushort row, ii, jj, ik;
    uint row_gid, kk;
    uint col_start, col_end, col_ptr, col_ptr_end;
    double val;

    for (row=0; row<10; row++) {
        row_gid = row*gsize + gid;       

        col_ptr = Anorm_diagptr[row];
        
        col_start = Anorm_colptr[col_ptr];
        col_end = Anorm_colptr[col_ptr + 1];
        
        val = 0.0;

        while (col_start < col_end) {
            ii = Anorm_indptr_i[col_start];
            jj = Anorm_indptr_j[col_start];
            ik = Anorm_indices[col_start];   
            kk = gsize*ik + gid;             
            val += Adata[ii]*Adata[jj]*x[kk]/z[kk];
            col_start += 1;
        }

        out[row_gid] = r[row_gid]/val;
    }
}

double dot_product(__global double* x, __global double* y, int N) {
    /* dot product of x and y */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    ushort row;
    uint row_gid;
    double val = 0.0;

    for (row=0; row<N; row++) {
        row_gid = row*gsize + gid;
        val += x[row_gid]*y[row_gid];
        //printf("dot prod %d:, %f %f %f\n", row_gid, x[row_gid], y[row_gid], val);
    }
    return val;
}

void vector_copy(__global double* x, __global double* y, int N) {
    /* y = x */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    ushort row;
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
    ushort row;
    uint row_gid;

    for (row=0; row<N; row++) {
        row_gid = row*gsize + gid;
        x[row_gid] = xscale*x[row_gid] + yscale*y[row_gid];
        //printf("update %d: %f %f %f %f\n", row_gid, xscale, x[row_gid], yscale, y[row_gid]);
    }
}

void vector_set(__global double* x, double scalar, int N) {
    /* x = scalar */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    ushort row;
    uint row_gid;

    for (row=0; row<N; row++) {
        row_gid = row*gsize + gid;
        x[row_gid] = scalar;
    }
}

double vector_max(__global double *x, int N) {
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    ushort row;
    uint row_gid;

    double val = -INFINITY;

    for (row=0; row<N; row++) {
        row_gid = row*gsize + gid;
        val = max(val, x[row_gid]);
    }
    return val;
}


__kernel void normal_eqn_conjugate_gradient(
    __constant double* Adata,
    ushort Asize,
    __constant ushort* Anorm_rowptr,
    __constant ushort* Anorm_diagptr,
    __constant ushort* Anorm_colptr,
    __constant ushort* Anorm_colindices,
    __constant ushort* Anorm_indices,
    __constant ushort* Anorm_indptr_i,
    __constant ushort* Anorm_indptr_j,
    __global double* x, __global double* z, __global double* b, __global double* y,
    __global double* r, __global double* p, __global double* w
) {
    /* Solve the normal equations for y

    (A(x/z)A^T)y = b

    */
    int gid = get_global_id(0);
    int iter;
    double r_z, r_z_next, alpha, rr, beta;

    // Compute the initial residuals
    residuals(Adata, Asize, Anorm_rowptr, Anorm_colptr, Anorm_colindices, Anorm_indices,
              Anorm_indptr_i, Anorm_indptr_j, x, z, b, y, r);    

    // Compute preconditioned residuals
    preconditioned_residuals(Adata, Asize, Anorm_diagptr, Anorm_colptr, Anorm_colindices, Anorm_indices,
                             Anorm_indptr_i, Anorm_indptr_j, x, z, r, w);

    r_z = dot_product(r, w, Asize);

    // initialise p
    vector_copy(w, p, Asize);

    for (iter=0; iter<20; iter++) {
        //printf("Iteration: %d\n", iter);
        alpha = r_z / vector_normal_eqn_vector_product(Adata, Asize, Anorm_rowptr, Anorm_colptr, Anorm_colindices, Anorm_indices,
                                                       Anorm_indptr_i, Anorm_indptr_j, x, z, p);
        //printf("Alpha: %g %g\n", r_z, alpha);
        // Update y
        vector_update(y, p, 1.0, alpha, Asize);

        // Update the residuals
        residuals(Adata, Asize, Anorm_rowptr, Anorm_colptr, Anorm_colindices, Anorm_indices,
              Anorm_indptr_i, Anorm_indptr_j, x, z, b, y, r);

        rr = sqrt(dot_product(r, r, Asize));

        printf("%d iter: %d: |residuals| %g\n", gid, iter, rr);

        if (rr < 1e-8) {
            //printf("%d iter: %d: |residuals| %g\n", gid, iter, rr);
            // printf("Solved!\n");
            break;
        }

        // Compute preconditioned residuals
        preconditioned_residuals(Adata, Asize, Anorm_diagptr, Anorm_colptr, Anorm_colindices, Anorm_indices,
                             Anorm_indptr_i, Anorm_indptr_j, x, z, r, w);

        r_z_next = dot_product(r, w, Asize);
        beta = r_z_next / r_z;

        vector_update(p, w, beta, 1.0, Asize);

        r_z = r_z_next;
    }
}


__kernel void normal_eqn_rhs(
    matrix(A),  // Sparse A matrix
    matrix(AT),  // Sparse transpose of A matrix
    __global double* x,
    __global double* z,
    __global double* b,
    __global double* c,
    __global double* y,
    double mu,
    __global double* tmp, // work array size of x
    __global double* out // work array size of b
) {
    /* Compute the right-hand side of the system of primal normal equations

    rhs = -(b - A.dot(x) - A.dot(x * (c - At.dot(y) + mu/x)/z))
    */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    ushort row, col, index, last_index;
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
        val = 0.0;

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
    __global double* x, __global double* b
) {
    /* Calculate primal-feasibility

        normr = b - A.dot(x)
    */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    ushort row, col, index, last_index;
    uint row_gid;
    double val;

    // Compute primal feasibility
    double normr = 0.0;
    for (row=0; row<Asize; row++) {
        row_gid = row*gsize + gid;
        val = b[row_gid];

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

        norms = c - AT.dot(y) - z
    */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    ushort row, col, index, last_index;
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

double compute_dx_dz(
    matrix(AT), // Sparse transpose of A matrix
    __global double* x, __global double* z, __global double* c, __global double* y, __global double* dy, double mu,
    __global double* dx, __global double* dz
) {
    /*

        dx = (c - AT.dot(y) - AT.dot(dy) + mu/x)*x/z
        dz = (mu - z*dx)/x - z
    */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    ushort row, col, index, last_index;
    uint row_gid;
    double val, val2;

    double theta = -INFINITY;

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

        theta = max(max(theta, -dx[row_gid]/x[row_gid]), -dz[row_gid]/z[row_gid]);
    }

    return theta;
}


ushort normal_eqn_step(
    matrix(A),  // Sparse A matrix
    matrix(AT),  // Sparse transpose of A matrix
    __constant ushort* Anorm_rowptr,
    __constant ushort* Anorm_diagptr,
    __constant ushort* Anorm_colptr,
    __constant ushort* Anorm_colindices,
    __constant ushort* Anorm_indices,
    __constant ushort* Anorm_indptr_i,
    __constant ushort* Anorm_indptr_j,
    __global double* x, __global double* z, __global double* b, __global double* c, __global double* y, double delta,
    __global double* dx, __global double* dz, __global double* dy,  // Work arrays for conjugate gradient method
    __global double* r, __global double* p, __global double* w,  // Work arrays for conjugate gradient method
    __global double* tmp, // work array size of x
    __global double* tmp2 // work array size of b
) {
    /* Perform a single step of the path-following algorithm.

    */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);

    // Compute feasibilities
    double normr = primal_feasibility(Aindptr, Aindices, Adata, Asize, x, b);
    double norms = dual_feasibility(ATindptr, ATindices, ATdata, ATsize, y, c, z);
    // Compute optimality
    double gamma = dot_product(z, x, ATsize);
    double mu = delta * gamma / ATsize;

    double max_x = vector_max(x, ATsize);
    double max_y = vector_max(y, Asize);

    printf("norm-r: %g, norm-s: %g, gamma: %g, max(x): %g, max(y): %g\n", normr, norms, gamma, max_x, max_y);
    if ((normr < 1e-6) && (norms < 1e-6) && (gamma < 1e-6)) {
        // Feasible and optimal; no further work!
        // TODO set a status output?
        return 0;
    }

    // Solve normal equations
    //   1. Calculate the RHS (into tmp2)
    normal_eqn_rhs(
        Aindptr, Aindices, Adata, Asize, ATindptr, ATindices, ATdata, ATsize,
        x, z, b, c, y, mu, tmp, tmp2
    );

    //   2. Set initial guess of dy
    vector_set(dy, 0.0, Asize);

    //   3. Solve the normal equations for dy
    normal_eqn_conjugate_gradient(
        Adata, Asize, Anorm_rowptr, Anorm_diagptr, Anorm_colptr, Anorm_colindices, Anorm_indices, Anorm_indptr_i, Anorm_indptr_j,
        x, z, tmp2, dy, r, p, w
    );

    // Calculate dx and dz
    //     dx = (c - AT.dot(y) - AT.dot(dy) + mu/x)*x/z
    //     dz = (mu - z*dx)/x - z
    double theta = compute_dx_dz(
        ATindptr, ATindices, ATdata, ATsize,
        x, z, c, y, dy, mu, dx, dz
    );

    theta = min(0.95/theta, 1.0);

    vector_update(x, dx, 1.0, theta, ATsize);
    vector_update(z, dz, 1.0, theta, ATsize);
    vector_update(y, dy, 1.0, theta, Asize);

    return 1;
}

__kernel void normal_eqn_solve(
    matrix(A),  // Sparse A matrix
    matrix(AT),  // Sparse transpose of A matrix
    __constant ushort* Anorm_rowptr,
    __constant ushort* Anorm_diagptr,
    __constant ushort* Anorm_colptr,
    __constant ushort* Anorm_colindices,
    __constant ushort* Anorm_indices,
    __constant ushort* Anorm_indptr_i,
    __constant ushort* Anorm_indptr_j,
    __global double* x,
    __global double* z,
    __global double* b,
    __global double* c,
    __global double* y,
    double delta,
    __global double* dx,
    __global double* dz,
    __global double* dy,  // Work arrays for conjugate gradient method
    __global double* r,
    __global double* p,
    __global double* w,  // Work arrays for conjugate gradient method
    __global double* tmp, // work array size of x
    __global double* tmp2, // work array size of b
    uint init
) {
    uint i;
    ushort status;
    //printf("Starting solve kernel ...");

    if (init == 1) {
        //printf("Resetting vectors ...");
        vector_set(x, 1.0, ATsize);
        vector_set(z, 1.0, ATsize);
        vector_set(y, 1.0, Asize);
    }

    for (i=0; i<10; i++) {
        status = normal_eqn_step(
            Aindptr, Aindices, Adata, Asize,
            ATindptr, ATindices, ATdata, ATsize,
            Anorm_rowptr, Anorm_diagptr, Anorm_colptr, Anorm_colindices, Anorm_indices, Anorm_indptr_i, Anorm_indptr_j,
            x, z, b, c, y, delta, dx, dz, dy,
            r, p, w, tmp, tmp2
        );
        if (status == 0) {
            return;
        }
    }
}

