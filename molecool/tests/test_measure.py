"""
Unit and regression test for the measure module.
"""

# Import package, test suite, and other packages as needed
import molecool
import numpy as np
import pytest

def test_calculate_angle_radians():
    """Test the calculate_angle function with output in radians"""

    r1 = np.array([1, 0, 0])
    r2 = np.array([0, 0, 0])
    r3 = np.array([0, 1, 0])

    expected_value = 90 * (np.pi / 180)

    calculated_value = molecool.calculate_angle(r1, r2, r3)
    assert expected_value == calculated_value


@pytest.mark.slow
def test_calculate_distance():
    """Test that calculate_distance function calculates what we expect."""
    
    r1 = np.array([0, 0, 0])
    r2 = np.array([0, 1, 0])

    expected_distance = 1

    calculated_distance = molecool.calculate_distance(r1, r2)

    assert expected_distance == calculated_distance

def test_calculate_angle():
   """Test the calculate_angle function"""

   r1 = np.array([1, 0, 0])
   r2 = np.array([0, 0, 0])
   r3 = np.array([0, 1, 0])

   expected_value = 90

   calculated_value = molecool.calculate_angle(r1, r2, r3, degrees=True)
   assert expected_value == calculated_value



