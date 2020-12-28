import pyopencl as cl
import os

CL_DIR = os.path.join(os.path.dirname(__file__), 'cl')


def get_cl_program(cl_context: cl.Context, fast_relaxed_math: bool = False) -> cl.Program:

    options = []
    if fast_relaxed_math:
        options.append('-cl-fast-relaxed-math')

    with open(os.path.join(CL_DIR, 'path_following.cl')) as fh:
        prg = cl.Program(cl_context, fh.read()).build(options)

    return prg
