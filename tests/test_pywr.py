import datetime
import os
from pywr.model import Model
from pywr.core import Scenario
from pywr.nodes import Input, Link, Output
from numpy.testing import assert_allclose
from cgpycl.solver import PathFollowingIndirectClSolver  # noqa
import pytest

# os.environ['PYWR_SOLVER'] = 'path-following-indirect-cl'


@pytest.fixture(scope="module", params=['path-following-indirect-cl', 'path-following-direct-cl'])
def pywr_solver(request):
    os.environ['PYWR_SOLVER'] = request.param


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


def test_run_simple1(pywr_solver):
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
    assert (model.timestepper.current.datetime - t0 == datetime.timedelta(0))
    # check the timestamp incremented
    model.step()
    assert (model.timestepper.current.datetime - t0 == datetime.timedelta(1))


@pytest.mark.parametrize('size', [1, 32, 64, 128])
def test_transfer(pywr_solver, size):
    """Test a simple transfer model. """

    model = Model()

    Scenario(model, name='test', size=size)

    supply1 = Input(model, name='supply-1', max_flow=5.0)
    wtw1 = Link(model, name='wtw-1')
    demand1 = Output(model, name='demand-1', max_flow=10.0, cost=-10.0)

    supply1.connect(wtw1)
    wtw1.connect(demand1)

    supply2 = Input(model, name='supply-2', max_flow=15.0)
    wtw2 = Link(model, name='wtw-2')
    demand2 = Output(model, name='demand-2', max_flow=10.0, cost=-10.0)

    supply2.connect(wtw2)
    wtw2.connect(demand2)

    transfer21 = Link(model, name='transfer-12', max_flow=2.0, cost=1.0)
    wtw2.connect(transfer21)
    transfer21.connect(wtw1)

    model.setup()
    model.step()

    assert_allclose(supply1.flow, [5.0] * size)
    assert_allclose(demand1.flow, [7.0] * size)
    assert_allclose(supply2.flow, [12.0] * size)
    assert_allclose(demand2.flow, [10.0] * size)
    assert_allclose(transfer21.flow, [2.0] * size)


@pytest.mark.parametrize('size', [1, 32, 64, 128])
def test_transfer2(pywr_solver, size):
    """Test a simple transfer model. """

    model = Model()

    Scenario(model, name='test', size=size)

    demands = [9.47, 7.65, 9.04, 9.56, 9.44]
    supply = [4.74, 6.59, 12.04, 8.81, 11.75]

    for i, (s, d) in enumerate(zip(supply, demands)):
        s = Input(model, name=f'supply-{i}', max_flow=s)
        lnk = Link(model, name=f'wtw-{i}')
        d = Output(model, name=f'demand-{i}', max_flow=d, cost=-10.0)

        s.connect(lnk)
        lnk.connect(d)

    transfer04 = Link(model, name='transfer-04', max_flow=15.36, cost=1.0)

    model.nodes['wtw-0'].connect(transfer04)
    transfer04.connect(model.nodes['wtw-4'])

    model.setup()
    model.step()

    expected_supply = [4.74, 6.59, 9.04, 8.81, 9.44]

    for i, expected in enumerate(expected_supply):
        assert_allclose(model.nodes[f'supply-{i}'].flow, [expected] * size)

    assert_allclose(transfer04.flow, [0.0] * size, atol=1e-8)
