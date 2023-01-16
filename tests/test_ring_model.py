import matplotlib.pyplot as plt
import numpy as np
import pytest
from pyroll.core import Profile

import pyroll.ring_model
from pyroll.ring_model import RingProfile

pyroll.ring_model.RING_COUNT = 3


def test_equivalent_radius():
    p: RingProfile | Profile = Profile.round(radius=9)

    assert np.isclose(p.equivalent_radius, 9, rtol=1e-3)


def test_rings():
    p: RingProfile | Profile = Profile.round(radius=9)

    assert np.allclose(p.rings, [1.5, 4.5, 7.5], rtol=1e-3)


def test_ring_boundaries():
    p: RingProfile | Profile = Profile.round(radius=9)

    assert np.allclose(p.ring_boundaries, [0, 3, 6, 9], rtol=1e-3)


@pytest.mark.parametrize(
    "p", [
        Profile.round(radius=10),
        Profile.square(side=10, corner_radius=1),
        Profile.box(height=10, width=5, corner_radius=1),
        Profile.diamond(height=5, width=10, corner_radius=1)
    ]
    )
def test_ring_contours(p: RingProfile | Profile):
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.subplots()

    ax.set_aspect("equal")

    for c in reversed(p.ring_contours):
        plt.fill(*c.xy)
    plt.show()
