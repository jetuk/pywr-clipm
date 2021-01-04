import pyopencl as cl
import os

CL_DIR = os.path.join(os.path.dirname(__file__), 'cl')


def get_cl_program(cl_context: cl.Context, fast_relaxed_math: bool = False, **kwargs) -> cl.Program:

    filename = kwargs.get('filename', 'path_following.cl')
    code = kwargs.get('code', None)

    options = []
    if fast_relaxed_math:
        options.append('-cl-fast-relaxed-math')

    with open(os.path.join(CL_DIR, 'common.cl')) as fh:
        common_code = fh.read()

    if code is None:
        with open(os.path.join(CL_DIR, filename)) as fh:
            code = fh.read()

    prg = cl.Program(cl_context, "\n".join([common_code, code])).build(options)
    return prg
