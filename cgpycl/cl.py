import pyopencl as cl
import os

CL_DIR = os.path.join(os.path.dirname(__file__), 'cl')


def get_cl_program(cl_context: cl.Context) -> cl.Program:

    with open(os.path.join(CL_DIR, 'path_following.cl')) as fh:
        prg = cl.Program(cl_context, fh.read()).build()

    return prg
