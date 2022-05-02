import numpy as np
import molecool
import pytest
import numpy as np
import os

@pytest.fixture
def methane_molecule():
    symbols = np.array(['C', 'H', 'H', 'H', 'H'])
    coordinates = np.array([
        [1, 1, 1],
        [2.4, 1, 1],
        [-0.4, 1, 1],
        [1, 1, 2.4],
        [1, 1, -0.4],
    ])
    
    return symbols, coordinates

@pytest.mark.parametrize("p1, p2, p3, expected_angle", [
    (np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0]), np.array([0, 0, 0]), np.array([1, 0, 0]), 45),
    (np.array([0, 0, -1]), np.array([0, 1, 0]), np.array([1, 0, 0]), 60  ),
    (np.array([np.sqrt(3)/2, (1/2), 0]), np.array([0, 0, 0]), np.array([1, 0, 0]), 30),
])
def test_calculate_angle_many(p1, p2, p3, expected_angle):

    calculated_angle = molecool.calculate_angle(p1, p2, p3, degrees=True)

    assert expected_angle == pytest.approx(calculated_angle), F'{calculated_angle} {expected_angle}'


def test_build_bond_list(methane_molecule):
    symbols, coordinates = methane_molecule

    bonds = molecool.build_bond_list(coordinates)

    assert len(bonds) == 4

    for atoms, bonds in bonds.items():
        assert bonds == 1.4

def test_molecular_mass(methane_molecule):
    symbols, coordinates = methane_molecule
    
    calculated_mass = molecool.calculate_molecular_mass(symbols)

    actual_mass = 16.04

    assert pytest.approx(actual_mass, abs=1e-2) == calculated_mass

def test_build_bond_list_failure(methane_molecule):
    symbols, coordinates = methane_molecule
    
    with pytest.raises(ValueError):
        molecool.build_bond_list(coordinates, min_bond=-1)

def test_center_of_mass(methane_molecule):
    symbols, coordinates = methane_molecule

    center_of_mass = molecool.calculate_center_of_mass(symbols, coordinates)

    expected_center = np.array([1,1,1])
    
    assert np.array_equal(center_of_mass, expected_center)

def test_move_methane(methane_molecule):
    symbols, coordinates = methane_molecule

    coordinates[0] += 5


