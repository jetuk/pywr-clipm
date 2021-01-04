import pytest
import pyopencl as cl


@pytest.fixture()
def cl_context() -> cl.Context:
    return cl.create_some_context()


@pytest.fixture()
def cl_queue(cl_context) -> cl.CommandQueue:
    return cl.CommandQueue(cl_context)
