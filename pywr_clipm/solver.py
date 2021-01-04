from pywr.core import ModelStructureError
from pywr._core import BaseInput, BaseOutput, BaseLink, Storage, VirtualStorage, AggregatedNode
from pywr.solvers import Solver, solver_registry
from scipy.sparse import coo_matrix, hstack, eye
import time
import numpy as np
from typing import List
from dataclasses import dataclass
from collections import defaultdict
import pyopencl as cl
from .sparse import create_sparse_normal_matrix_buffers, create_sparse_normal_matrix_indices,\
    create_sparse_matrix_buffer, create_sparse_normal_matrix_cholesky_indices, \
    create_sparse_normal_matrix_cholesky_buffers
from .cl import get_cl_program


@dataclass
class AbstractNodeData:
    id: int = 0
    is_link: bool = False
    in_edges = []
    out_edges = []


class LP:
    def __init__(self):
        self._row_indices: List[int] = []
        self._col_indices: List[int] = []
        self._data: List[float] = []

    def add_entry(self, row: int, col: int, value: float):
        self._row_indices.append(row)
        self._col_indices.append(col)
        self._data.append(value)

    def add_row(self, cols: List[int], values: List[float], row: int = None) -> int:
        if len(cols) != len(values):
            raise ValueError(f'The number of columns {len(cols)} must match the number of '
                             f'values ({len(values)}).')
        if row is None:
            row = self.next_row
        for col, value in zip(cols, values):
            self.add_entry(row, col, value)
        return row

    @property
    def next_row(self) -> int:
        """Return the next row id"""
        return self.nrows

    @property
    def data(self):
        return self._data

    @property
    def row_indices(self):
        return self._row_indices

    @property
    def col_indices(self):
        return self._col_indices

    @property
    def nrows(self):
        if len(self.row_indices):
            n = max(self.row_indices) + 1
        else:
            n = 0
        return n

    @property
    def ncols(self):
        if len(self.row_indices):
            n = max(self.col_indices) + 1
        else:
            n = 0
        return n


class BasePathFollowingClSolver(Solver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cl_context = cl.create_some_context()
        self.cl_queue = cl.CommandQueue(self.cl_context)

    def setup(self, model):
        self.all_nodes = list(sorted(model.graph.nodes(), key=lambda n: n.fully_qualified_name))
        self.all_edges = edges = list(model.graph.edges())
        if not self.all_nodes or not self.all_edges:
            raise ModelStructureError("Model is empty")

        n = 0
        node_data = {}
        for _node in self.all_nodes:
            d = AbstractNodeData()
            d.id = n
            d.in_edges = []
            d.out_edges = []
            if isinstance(_node, BaseLink):
                d.is_link = True
            node_data[_node] = d
            n += 1
        self.num_nodes = n

        # Find cross-domain routes
        cross_domain_routes = model.find_all_routes(BaseOutput, BaseInput, max_length=2, domain_match='different')

        # create a lookup for the cross-domain routes.
        cross_domain_cols = {}
        for cross_domain_route in cross_domain_routes:
            # These routes are only 2 nodes. From output to input
            output, input = cross_domain_route
            # note that the conversion factor is not time varying
            conv_factor = input.get_conversion_factor()
            input_cols = [(n, conv_factor) for n in node_data[input].out_edges]
            # create easy lookup for the route columns this output might
            # provide cross-domain connection to
            if output in cross_domain_cols:
                cross_domain_cols[output].extend(input_cols)
            else:
                cross_domain_cols[output] = input_cols

        link_nodes = []
        non_storages = []
        storages = []
        virtual_storages = []
        aggregated_with_factors = []
        aggregated = []

        for some_node in self.all_nodes:
            if isinstance(some_node, (BaseInput, BaseLink, BaseOutput)):
                non_storages.append(some_node)
                if isinstance(some_node, BaseLink):
                    link_nodes.append(some_node)
            elif isinstance(some_node, VirtualStorage):
                virtual_storages.append(some_node)
            elif isinstance(some_node, Storage):
                storages.append(some_node)
            elif isinstance(some_node, AggregatedNode):
                if some_node.factors is not None:
                    aggregated_with_factors.append(some_node)
                aggregated.append(some_node)

        if len(non_storages) == 0:
            raise ModelStructureError("Model has no non-storage nodes")

        # Create a container for the inequality and equality constraints separately.
        self.num_edges = len(edges)
        lp_ineq = LP()
        lp_eq = LP()

        # create a lookup for edges associated with each node (ignoring cross domain edges)
        for row, (start_node, end_node) in enumerate(self.all_edges):
            if start_node.domain != end_node.domain:
                continue
            node_data[start_node].out_edges.append(row)
            node_data[end_node].in_edges.append(row)
            # print(row, start_node, end_node)

        # Apply nodal flow constraints
        self.row_map_ineq_non_storage = {}
        self.row_map_eq_non_storage = {}
        self.row_map_ineq_storage = {}

        for some_node in non_storages:
            # Don't add un-constrained nodes to the matrix
            if some_node.max_flow == np.inf:
                continue

            # Differentiate betwen the node type.
            # Input and other nodes use the outgoing edge flows to apply the flow constraint on
            # This requires the mass balance constraints to ensure the inflow and outflow are equal
            # The Output nodes, in contrast, apply the constraint to the incoming flow (because there is no out
            # going flow)
            d = node_data[some_node]

            if isinstance(some_node, BaseInput):
                cols = d.out_edges
                assert len(d.in_edges) == 0
            elif isinstance(some_node, BaseOutput):
                cols = d.in_edges
                assert len(d.out_edges) == 0
            else:
                # Other nodes apply their flow constraints to all routes passing through them
                cols = d.out_edges

            if some_node.min_flow == some_node.max_flow:
                # This is an equality constraint
                row = lp_eq.add_row(cols, [1.0 for _ in cols])
                self.row_map_eq_non_storage[some_node] = row
            else:
                if some_node.min_flow != 0.0:
                    raise NotImplementedError('Path following solver does not support non-zero '
                                              'minimum flows ({}).'.format(some_node.name))
                # Add an in-equality constraint for the max flow
                row = lp_ineq.add_row(cols, [1.0 for _ in cols])
                self.row_map_ineq_non_storage[some_node] = row
                # print(some_node, row, cols)

            # Add constraint for cross-domain routes
            # i.e. those from a demand to a supply
            if some_node in cross_domain_cols:
                col_vals = cross_domain_cols[some_node]

                row = lp_eq.add_row(
                    cols + [c for c, _ in col_vals],
                    [-1.0 for _ in cols] + [1. / v for _, v in col_vals]
                )

        # Add mass balance constraints
        for some_node in link_nodes:
            d = node_data[some_node]
            in_cols = d.in_edges
            out_cols = d.out_edges

            lp_eq.add_row(
                in_cols + out_cols,
                [1.0 for _ in in_cols] + [-1.0 for _ in out_cols]
            )

        # Add storage constraints
        for storage in storages:

            cols_output = []
            for output in storage.outputs:
                cols_output.extend(node_data[output].in_edges)
            cols_input = []
            for input in storage.inputs:
                cols_input.extend(node_data[input].out_edges)

            # Two rows needed for the range constraint on change in storage volume
            row1 = lp_ineq.add_row(
                cols_output + cols_input,
                [1.0 for _ in cols_output] + [-1.0 for _ in cols_input]
            )
            row2 = lp_ineq.add_row(
                cols_output + cols_input,
                [-1.0 for _ in cols_output] + [1.0 for _ in cols_input]
            )

            self.row_map_ineq_storage[storage] = (row1, row2)

        nscenarios = len(model.scenarios.combinations)

        self.num_inequality_constraints = lp_ineq.nrows

        self.a = coo_matrix(
            (lp_ineq.data + lp_eq.data,
             (
                 lp_ineq.row_indices + [r + lp_ineq.nrows for r in lp_eq.row_indices],
                 lp_ineq.col_indices + lp_eq.col_indices
             ))
        ).tocsr()

        # Add slacks to the inequality section
        self.a = hstack([self.a, eye(self.a.shape[0], lp_ineq.nrows)]).tocsr()

        from sksparse import cholmod

        factors = cholmod.cholesky_AAt(self.a, ordering_method='natural')
        L, D = factors.L_D()
        print('A', self.a.getnnz())
        print('L', L.getnnz())
        print('D', D.getnnz())

        # assert False

        self.b = np.zeros((self.a.shape[0], nscenarios))
        self.c = np.zeros((self.a.shape[1], nscenarios))

        self.edge_cost_arr = np.zeros((len(self.all_edges), nscenarios))
        self.edge_flows_arr = np.zeros((len(self.all_edges), nscenarios))
        self.node_flows_arr = np.zeros((self.num_nodes, nscenarios))

        self.node_data = node_data
        self.non_storages = non_storages
        self.storages = storages
        self.virtual_storages = virtual_storages
        self.aggregated = aggregated

        self._create_cl_buffers()

        # reset stats
        self._stats = defaultdict(lambda: 0.0, **{
            'total_cl': 0.0,
            'number_of_steps': 0,
            'number_of_solves': 0,
            'number_of_rows': self.a.shape[0],
            'number_of_cols': self.a.shape[1],
            'number_of_nonzero': self.a.getnnz(),
            'number_of_nodes': len(self.all_nodes)
        })

    def _create_cl_buffers(self):
        raise NotImplementedError()

    @property
    def stats(self):
        return self._stats

    def reset(self):
        pass

    def _update_c(self):
        edges = self.all_edges
        nedges = len(edges)
        c = self.c

        # Initialise the cost on each edge to zero
        edge_costs = self.edge_cost_arr
        for col in range(nedges):
            edge_costs[...] = 0.0

        # update the cost of each node in the model
        for _node in self.all_nodes:
            cost = np.array(_node.get_all_cost())
            data = self.node_data[_node]

            if data.is_link:
                cost /= 2

            # print(_node, cost, data.is_link, data.in_edges, data.out_edges)

            for col in data.in_edges:
                edge_costs[col, :] += cost
            for col in data.out_edges:
                edge_costs[col, :] += cost

        c[:nedges, :] = -edge_costs

    def _update_b(self, model):
        timestep = model.timestep

        b = self.b

        # update non-storage properties
        row = 0

        # Ineqality constraints come first with zero offset
        for node, row in self.row_map_ineq_non_storage.items():
            node.get_all_max_flow(out=b[row, :])
            # print(node, np.max(np.abs(b[row, :])), node.max_flow)

        # ... then equality constraints
        offset = self.num_inequality_constraints
        for node, row in self.row_map_eq_non_storage.items():
            node.get_all_max_flow(out=b[offset + row, :])
            # print(node, np.max(np.abs(b[offset + row, :])))

        # Non-storage constraints
        for storage, (row1, row2) in self.row_map_ineq_storage.items():
            max_volume = storage.get_all_max_volume()
            min_volume = storage.get_all_min_volume()
            volume = storage.volume

            avail_volume = volume - min_volume
            avail_volume[avail_volume < 0.0] = 0.0

            # change in storage cannot be more than the current volume or
            # result in maximum volume being exceeded
            lb = avail_volume / timestep.days
            ub = (max_volume - volume) / timestep.days
            ub[ub < 0.0] = 0.0

            b[row1, :] = ub
            b[row2, :] = lb

    def _update_flows(self):
        edges = self.all_edges
        edge_flows = self.edge_flows_arr

        # collect the total flow via each node
        node_flows = self.node_flows_arr
        node_flows[:] = 0.0
        for n, edge in enumerate(edges):
            flow = edge_flows[n, :]
            # print(n, edge, flow)
            for _node in edge:
                data = self.node_data[_node]
                if data.is_link:
                    node_flows[data.id, :] += flow / 2
                else:
                    node_flows[data.id, :] += flow

        # commit the total flows
        for n in range(0, self.num_nodes):
            _node = self.all_nodes[n]
            _node.commit_all(node_flows[n, :])


class PathFollowingIndirectClSolver(BasePathFollowingClSolver):
    name = 'path-following-indirect-cl'

    def _create_cl_buffers(self):
        """Create the OpenCL buffers for the solver. """
        MF = cl.mem_flags

        # Copy the sparse matrix, it's normal indices  and vector to the context
        self.a_buf = create_sparse_matrix_buffer(self.cl_context, self.a)
        self.at_buf = create_sparse_matrix_buffer(self.cl_context, self.a.T.tocsr())

        norm_indices = create_sparse_normal_matrix_indices(self.a)
        self.norm_a_buf = create_sparse_normal_matrix_buffers(self.cl_context, norm_indices)

        self.b_buf = cl.Buffer(self.cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=self.b)
        self.c_buf = cl.Buffer(self.cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=self.c)

        self.x = np.zeros_like(self.c)
        self.x_buf = cl.Buffer(self.cl_context, MF.READ_WRITE, self.c.nbytes)
        self.z_buf = cl.Buffer(self.cl_context, MF.READ_WRITE, self.c.nbytes)
        self.y_buf = cl.Buffer(self.cl_context, MF.READ_WRITE, self.b.nbytes)
        # TODO reduce the size of this
        self.w_buf = cl.Buffer(self.cl_context, MF.READ_WRITE, self.b.nbytes)

        self.dx_buf = cl.Buffer(self.cl_context, MF.READ_WRITE, self.c.nbytes)
        self.dz_buf = cl.Buffer(self.cl_context, MF.READ_WRITE, self.c.nbytes)
        self.dy_buf = cl.Buffer(self.cl_context, MF.READ_WRITE, self.b.nbytes)
        # TODO reduce the size of this
        self.dw_buf = cl.Buffer(self.cl_context, MF.READ_WRITE, self.b.nbytes)

        # Work arrays
        self.r_buf = cl.Buffer(self.cl_context, MF.WRITE_ONLY, self.b.nbytes)
        self.p_buf = cl.Buffer(self.cl_context, MF.WRITE_ONLY, self.b.nbytes)
        self.s_buf = cl.Buffer(self.cl_context, MF.WRITE_ONLY, self.b.nbytes)

        self.tmp_buf = cl.Buffer(self.cl_context, MF.WRITE_ONLY, self.c.nbytes)
        self.rhs_buf = cl.Buffer(self.cl_context, MF.WRITE_ONLY, self.b.nbytes)

        self.program = get_cl_program(self.cl_context)

    def solve(self, model):

        self._update_b(model)
        self._update_c()

        # print(self.a.shape, self.num_inequality_constraints)
        # print('a', self.a)
        # print('b', self.b)
        # print('c', self.c)

        # TODO can b be copied to device while c is updated on host?
        cl.enqueue_copy(self.cl_queue, self.b_buf, self.b)
        cl.enqueue_copy(self.cl_queue, self.c_buf, self.c)

        a_buf = self.a_buf
        at_buf = self.at_buf
        norm_a_buf = self.norm_a_buf

        print('Starting solve ...')
        t0 = time.perf_counter()
        self.program.normal_eqn_solve(
            self.cl_queue, (self.b.shape[1],), None,
            a_buf.indptr, a_buf.indices, a_buf.data, a_buf.nrows,
            at_buf.indptr, at_buf.indices, at_buf.data, at_buf.nrows,
            norm_a_buf.rowptr, norm_a_buf.diagptr, norm_a_buf.colptr,
            norm_a_buf.colindices, norm_a_buf.indices, norm_a_buf.indptr1, norm_a_buf.indptr2,
            self.x_buf,
            self.z_buf,
            self.y_buf,
            self.w_buf,
            np.uint32(self.num_inequality_constraints),
            self.b_buf,
            self.c_buf,
            np.float64(0.02),
            self.dx_buf,
            self.dz_buf,
            self.dy_buf,
            self.dw_buf,
            self.r_buf,
            self.p_buf,
            self.s_buf,
            self.tmp_buf,
            self.rhs_buf,
            np.uint32(1)
        )

        cl.enqueue_copy(self.cl_queue, self.x, self.x_buf)
        # print('x', self.x)
        print(f'Solve completed in {time.perf_counter() - t0}s')

        if np.any(np.isnan(self.x)):
            raise RuntimeError('NaNs in solution!')

        self.edge_flows_arr[...] = self.x[:self.num_edges, :]
        self._update_flows()


solver_registry.append(PathFollowingIndirectClSolver)


class PathFollowingDirectClSolver(BasePathFollowingClSolver):
    name = 'path-following-direct-cl'

    def _create_cl_buffers(self):
        """Create the OpenCL buffers for the solver. """
        MF = cl.mem_flags

        # Copy the sparse matrix, it's normal indices  and vector to the context
        self.a_buf = create_sparse_matrix_buffer(self.cl_context, self.a)
        self.at_buf = create_sparse_matrix_buffer(self.cl_context, self.a.T.tocsr())

        cholesky_indices = create_sparse_normal_matrix_cholesky_indices(self.a)
        self.cholesky_buf = create_sparse_normal_matrix_cholesky_buffers(self.cl_context, cholesky_indices)

        gsize = self.b.shape[1]
        self.ldata = np.zeros((cholesky_indices.lindices.shape[0], gsize))
        print(self.ldata.shape)
        self.ldata_buf = cl.Buffer(self.cl_context, MF.WRITE_ONLY, self.ldata.nbytes)

        self.b_buf = cl.Buffer(self.cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=self.b)
        self.c_buf = cl.Buffer(self.cl_context, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=self.c)

        self.x = np.zeros_like(self.c)
        self.x_buf = cl.Buffer(self.cl_context, MF.READ_WRITE, self.c.nbytes)
        self.z_buf = cl.Buffer(self.cl_context, MF.READ_WRITE, self.c.nbytes)
        self.y_buf = cl.Buffer(self.cl_context, MF.READ_WRITE, self.b.nbytes)
        # TODO reduce the size of this
        self.w_buf = cl.Buffer(self.cl_context, MF.READ_WRITE, self.b.nbytes)

        self.dx_buf = cl.Buffer(self.cl_context, MF.READ_WRITE, self.c.nbytes)
        self.dz_buf = cl.Buffer(self.cl_context, MF.READ_WRITE, self.c.nbytes)
        self.dy_buf = cl.Buffer(self.cl_context, MF.READ_WRITE, self.b.nbytes)
        # TODO reduce the size of this
        self.dw_buf = cl.Buffer(self.cl_context, MF.READ_WRITE, self.b.nbytes)

        # Work arrays
        self.tmp_buf = cl.Buffer(self.cl_context, MF.WRITE_ONLY, self.c.nbytes)
        self.rhs_buf = cl.Buffer(self.cl_context, MF.WRITE_ONLY, self.b.nbytes)

        self.status = np.zeros(gsize, dtype=np.uint32)
        self.status_buf = cl.Buffer(self.cl_context, MF.WRITE_ONLY, self.status.nbytes)

        self.program = get_cl_program(self.cl_context, filename='path_following_direct.cl')

    def solve(self, model):

        self._update_b(model)
        self._update_c()

        # print(self.a.shape, self.num_inequality_constraints)
        # print('a', self.a)
        # print('b', self.b)
        # print('c', self.c)

        # TODO can b be copied to device while c is updated on host?
        cl.enqueue_copy(self.cl_queue, self.b_buf, self.b)
        cl.enqueue_copy(self.cl_queue, self.c_buf, self.c)

        a_buf = self.a_buf
        at_buf = self.at_buf
        cholesky_buf = self.cholesky_buf

        print('Starting solve ...')
        t0 = time.perf_counter()

        self.program.normal_eqn_init(
            self.cl_queue, (self.b.shape[1],), None,
            a_buf.nrows,
            at_buf.nrows,
            self.x_buf,
            self.z_buf,
            self.y_buf,
            self.w_buf,
            np.uint32(self.num_inequality_constraints),
        )

        for iter in range(200):
            self.program.normal_eqn_step(
                self.cl_queue, (self.b.shape[1],), None,
                a_buf.indptr, a_buf.indices, a_buf.data, a_buf.nrows,
                at_buf.indptr, at_buf.indices, at_buf.data, at_buf.nrows,
                cholesky_buf.anorm_indptr, cholesky_buf.anorm_indptr_i, cholesky_buf.anorm_indptr_j,
                cholesky_buf.anorm_indices,
                cholesky_buf.ldecomp_indptr, cholesky_buf.ldecomp_indptr_i, cholesky_buf.ldecomp_indptr_j,
                cholesky_buf.lindptr, cholesky_buf.ldiag_indptr, cholesky_buf.lindices,
                cholesky_buf.ltindptr, cholesky_buf.ltindices, cholesky_buf.ltmap,
                self.ldata_buf,
                self.x_buf,
                self.z_buf,
                self.y_buf,
                self.w_buf,
                np.uint32(self.num_inequality_constraints),
                self.b_buf,
                self.c_buf,
                np.float64(0.02),
                self.dx_buf,
                self.dz_buf,
                self.dy_buf,
                self.dw_buf,
                self.tmp_buf,
                self.rhs_buf,
                self.status_buf,
            )
            cl.enqueue_copy(self.cl_queue, self.status, self.status_buf)
            if np.all(self.status == 0):
                break
        if not np.all(self.status == 0):
            print(self.status)
            print(np.sum(self.status))
            print(np.where(self.status == 1))
            raise RuntimeError('Failed to solve!')

        cl.enqueue_copy(self.cl_queue, self.x, self.x_buf)
        # print('x', self.x)
        print(f'Solve completed in {time.perf_counter() - t0}s')

        if np.any(np.isnan(self.x)):
            raise RuntimeError('NaNs in solution!')

        self.edge_flows_arr[...] = self.x[:self.num_edges, :]
        self._update_flows()


solver_registry.append(PathFollowingDirectClSolver)
