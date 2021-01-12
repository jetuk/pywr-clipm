import pyopencl as cl
import os
from typing import Optional

CL_DIR = os.path.join(os.path.dirname(__file__))


def get_cl_program(cl_context: cl.Context, filename: str = 'path_following.cl', code: Optional[str] = None,
                   debug_gid: Optional[int] = None, fast_relaxed_math: bool = False) -> cl.Program:
    """Compile the OpenCL program

    Parameters
    ==========
    cl_context : pyopencl.Context
        OpenCL context in which to compile the program.
    filename : str
        Filename of the program to compile.
    code : Optional[str]
        A string of OpenCL code to compile instead of reading from a file.
    debug_gid : Optional[int]
        Optional global work id to print debugging output for.
    fast_relaxed_math : bool
        Whether to compiler with -cl-fast-relaxed-math or not (default False).

    """

    options = []
    if fast_relaxed_math:
        options.append('-cl-fast-relaxed-math')

    if debug_gid is not None:
        options.append(f'-DDEBUG_GID={debug_gid}')

    with open(os.path.join(CL_DIR, 'common.cl')) as fh:
        common_code = fh.read()

    if code is None:
        with open(os.path.join(CL_DIR, filename)) as fh:
            code = fh.read()

    prg = cl.Program(cl_context, "\n".join([common_code, code])).build(options)
    return prg
