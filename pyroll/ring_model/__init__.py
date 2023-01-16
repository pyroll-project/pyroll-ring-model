import numpy as np
import shapely

VERSION = "2.0.0b"

from pyroll.core import Profile, Hook

RING_COUNT = 5


@Profile.extension_class
class RingProfile(Profile):
    equivalent_radius = Hook[float]()
    """Radius of an equivalent round profile."""

    rings = Hook[np.ndarray]()
    """Array of radius coordinates core to surface."""

    ring_boundaries = Hook[np.ndarray]()
    """Array of radius coordinates core to surface."""

    ring_contours = Hook[np.ndarray]()
    """Array of radius coordinates core to surface."""


@RingProfile.equivalent_radius
def equivalent_radius(self: RingProfile):
    return np.sqrt(self.cross_section.area / np.pi)


@RingProfile.rings
def rings(self: RingProfile):
    return (self.ring_boundaries[1:] + self.ring_boundaries[:-1]) / 2


@RingProfile.ring_boundaries
def ring_boundaries(self: RingProfile):
    return np.linspace(0, self.equivalent_radius, RING_COUNT + 1)


@RingProfile.ring_contours
def ring_contours(self: RingProfile):
    a = np.zeros(len(self.rings) + 1, dtype=object)
    a[0] = shapely.LinearRing()
    x, y = np.array(self.cross_section.boundary.xy)
    angles = np.arctan2(y, x)
    radii = np.sqrt(x ** 2 + y ** 2)
    for i in range(1, len(a)):
        scaled_radii = radii * self.ring_boundaries[i] / self.equivalent_radius
        a[i] = shapely.LinearRing(np.column_stack([
            scaled_radii * np.cos(angles),
            scaled_radii * np.sin(angles)
        ]))

    return a
