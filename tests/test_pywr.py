import datetime
import os
from pywr.model import Model
from numpy.testing import assert_allclose
from cgpycl.solver import PathFollowingClSolver


# os.environ['PYWR_SOLVER'] = 'affine_scaling_cl'
os.environ['PYWR_SOLVER'] = 'path-following-cl'

def load_model(filename=None, data=None, solver=None):
    '''Load a test model and check it'''
    if data is None:
        path = os.path.join(os.path.dirname(__file__), 'models')
        with open(os.path.join(path, filename), 'r') as f:
            data = f.read()
    else:
        path = None

    model = Model.loads(data, path=path, solver=solver)
    model.check()
    return model


def test_run_simple1():
    '''Test the most basic model possible'''
    # parse the JSON into a model
    model = load_model('simple1.json')

    # run the model
    t0 = model.timestepper.start.to_pydatetime()
    model.step()

    # check results
    demand1 = model.nodes['demand1']
    assert_allclose(demand1.flow, 10.0, atol=1e-6, rtol=1e-6)
    # initially the timestepper returns the first time-step, so timestepper.current
    # does not change after the first 'step'.
    assert(model.timestepper.current.datetime - t0 == datetime.timedelta(0))
    # check the timestamp incremented
    model.step()
    assert(model.timestepper.current.datetime - t0 == datetime.timedelta(1))

