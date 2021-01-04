


__kernel void normal_matrix_cholesky_decomposition(
    __constant double* Adata, uint Asize,
    __constant uint* Anorm_indptr,
    __constant uint* Anorm_indptr_i,
    __constant uint* Anorm_indptr_j,
    __constant uint* Anorm_indices,
    __constant uint* Ldecomp_indptr,
    __constant uint* Ldecomp_indptr_i,
    __constant uint* Ldecomp_indptr_j,
    __global double* x,
    __global double* z,
    __global double* y,
    __global double* w,
    uint wsize,    
    __constant uint* Lindptr,
    __constant uint* Ldiag_indptr,
    __constant uint* Lindices,
    __global double* Ldata
) {
    /* 
    */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);  
    uint row, col, ii, jj;
    uint row_gid, xind;
    uint row_ind, row_ind_end;
    uint ind, ind_end;
    double val, inner_val;

    uint Lentry = 0;

    for (row = 0; row<Asize; row++) {
        row_gid = row*gsize + gid;

        row_ind = Lindptr[row];
        row_ind_end = Lindptr[row + 1];

        // Iterate the columns of L
        for (; row_ind < row_ind_end; row_ind++) {
            col = Lindices[row_ind];

            // Compute the normal equation element AAT[i, j]
            val = 0.0;
            ind = Anorm_indptr[Lentry];
            ind_end = Anorm_indptr[Lentry + 1];

            for (; ind < ind_end; ind++) {
                xind = Anorm_indices[ind] * gsize + gid;
                val += Adata[Anorm_indptr_i[ind]] * Adata[Anorm_indptr_j[ind]] * x[xind] / z[xind];
            }
            // Now remove the previous L entries
            ind = Ldecomp_indptr[Lentry];
            ind_end = Ldecomp_indptr[Lentry + 1];

            for (; ind < ind_end; ind++) {
                val -= Ldata[Ldecomp_indptr_i[ind] * gsize + gid] * Ldata[Ldecomp_indptr_j[ind] * gsize + gid];

            }

            if (row == col) {
                if (row < wsize) {
                    val += w[row_gid] / y[row_gid];
                }          
                val = sqrt(fabs(val));  
            } else {
                val = val / Ldata[Ldiag_indptr[col]*gsize + gid];
            }
            Ldata[Lentry * gsize + gid] = val;
            Lentry++;
        }
    }
}

__kernel void cholesky_solve(
    uint Asize,
    __constant uint* Lindptr,
    __constant uint* Lindices,
    __constant uint* LTindptr,
    __constant uint* LTindices,
    __constant uint* LTmap,
    __global double* Ldata,
    __global double* b,
    __global double* x
) {
  /* Solve a system Ax = b for x given the decomposition of A as L.

  L is a lower triangular matrix. Entries are stored such that the lth
  entry of L is the i(i + 1)/2 + j entry in dense i, j  coordinates.
  */
  int i, j, jk, jkk;
  uint gid = get_global_id(0);
  uint gsize = get_global_size(0);

  // Forward substitution
  for (i=0; i<Asize; i++) {
    
    x[i*gsize+gid] = b[i*gsize+gid];

    jk = Lindptr[i];
    j = Lindices[jk];

    while (j < i) {
      x[i*gsize+gid] -= x[j*gsize+gid]*Ldata[jk*gsize+gid];
      jk += 1;
      j = Lindices[jk];
    }
    // jk should now point to the (i, i) entry.
    x[i*gsize+gid] /= Ldata[jk*gsize+gid];
  }

  // Backward substitution
  for (i=Asize-1; i>=0; i--) {
    ///printf("%d %d\n", i, Asize);

    jk = LTindptr[i]+1;
    jkk = LTindptr[i+1];
    j = LTindices[jk];

    while(jk < jkk) {
        x[i*gsize+gid] -= x[j*gsize+gid]*Ldata[LTmap[jk]*gsize+gid];
        jk += 1;
        j = LTindices[jk];
    }

    jk = Lindptr[i+1]-1;
    x[i*gsize+gid] /= Ldata[jk*gsize+gid];
  }
}

__kernel void normal_eqn_step(
    matrix(A),  // Sparse A matrix
    matrix(AT),  // Sparse transpose of A matrix
    __constant uint* Anorm_indptr,
    __constant uint* Anorm_indptr_i,
    __constant uint* Anorm_indptr_j,
    __constant uint* Anorm_indices,
    __constant uint* Ldecomp_indptr,
    __constant uint* Ldecomp_indptr_i,
    __constant uint* Ldecomp_indptr_j,
    __constant uint* Lindptr,
    __constant uint* Ldiag_indptr,
    __constant uint* Lindices,
    __constant uint* LTindptr,    
    __constant uint* LTindices,    
    __constant uint* LTmap,    
    __global double* Ldata,
    __global double* x,
    __global double* z,
    __global double* y,
    __global double* w,
    uint wsize,    
    __global double* b,
    __global double* c,
    double delta,
    __global double* dx,
    __global double* dz,
    __global double* dy,
    __global double* dw,
    __global double* tmp, 
    __global double* tmp2,
    __global uint* status
) {
    /* Perform a single step of the path-following algorithm.

    */
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);

    // Compute feasibilities
    double normr = primal_feasibility(Aindptr, Aindices, Adata, Asize, x, w, wsize, b);
    double norms = dual_feasibility(ATindptr, ATindices, ATdata, ATsize, y, c, z);
    // Compute optimality
    double gamma = dot_product(z, x, ATsize) + dot_product(w, y, wsize);
    double mu = delta * gamma / (ATsize + wsize);

    double max_x = vector_max(x, ATsize);
    double max_y = vector_max(y, Asize);

    if (gid == 2) {
        printf("%d %d norm-r: %g, norm-s: %g, gamma: %g, max(x): %g, max(y): %g\n", gid, wsize, normr, norms, gamma, max_x, max_y);
    }
    if ((normr < 1e-6) && (norms < 1e-6) && (gamma < 1e-6)) {
        // Feasible and optimal; no further work!
        status[gid] = 0;
        return;
    }

    // Solve normal equations
    //   1. Calculate the RHS (into tmp2)
    normal_eqn_rhs(
        Aindptr, Aindices, Adata, Asize, ATindptr, ATindices, ATdata, ATsize,
        x, z, y, b, c, mu, wsize, tmp, tmp2
    );

    //   2. Compute decomposition of normal matrix
    normal_matrix_cholesky_decomposition(
        Adata,
        Asize,
        Anorm_indptr,
        Anorm_indptr_i,
        Anorm_indptr_j,
        Anorm_indices,
        Ldecomp_indptr,
        Ldecomp_indptr_i,
        Ldecomp_indptr_j,
        x,
        z,
        y,
        w,
        wsize,    
        Lindptr,
        Ldiag_indptr,
        Lindices,
        Ldata
    );

    //   3. Solve system directly
    cholesky_solve(
        Asize,
        Lindptr,
        Lindices,
        LTindptr,
        LTindices,
        LTmap,
        Ldata,
        tmp2,
        dy
    );

    // Calculate dx and dz
    //     dx = (c - AT.dot(y) - AT.dot(dy) + mu/x)*x/z
    //     dz = (mu - z*dx)/x - z
    //     dw = (mu - w*dy)/y - w
    double theta = compute_dx_dz_dw(
        Asize, ATindptr, ATindices, ATdata, ATsize,
        x, z, y, w, wsize, c, dy, mu, dx, dz, dw
    );

    theta = min(0.9/theta, 1.0);
    // if (gid == 0) {
    //     printf("%d theta: %g", gid, theta);
    // }

    vector_update(x, dx, 1.0, theta, ATsize);
    vector_update(z, dz, 1.0, theta, ATsize);
    vector_update(y, dy, 1.0, theta, Asize);
    vector_update(w, dw, 1.0, theta, wsize);

    status[gid] = 1;
}

__kernel void normal_eqn_init(
    uint Asize,
    uint ATsize,
    __global double* x,
    __global double* z,
    __global double* y,
    __global double* w,
    uint wsize
) {
    vector_set(x, 1.0, ATsize);
    vector_set(z, 1.0, ATsize);
    vector_set(y, 1.0, Asize);
    vector_set(w, 1.0, wsize);
}
