import numpy as np

class Aggregate:
    """Core physical entity representing a fractal cluster.

    Uses pre-allocated contiguous memory via NumPy for high data locality 
    and access performance.
    """
    def __init__(self, max_particles: int, length_unit: str = 'nm', mass_unit: str = 'g', density: float = 1.0):
        """Initializes the Aggregate object.

        Args:
            max_particles (int): Maximum number of particles the cluster can hold.
            length_unit (str, optional): Unit for length measurements (e.g., 'nm'). Defaults to 'nm'.
            mass_unit (str, optional): Unit for mass measurements (e.g., 'g'). Defaults to 'g'.
            density (float, optional): Density of particle material. Defaults to 1.0.

        Raises:
            ValueError: If max_particles <= 0.
        """
        if max_particles <= 0:
            raise ValueError("max_particles must be positive")
        
        # Data structure: [x, y, z, radius, mass] with pre-allocated memory
        self._data = np.zeros((max_particles, 5), dtype=np.float64)
        self._current_size = 0  
        self.length_unit = length_unit
        self.mass_unit = mass_unit
        self.density = density

    @property
    def positions(self) -> np.ndarray:
        """Gets particle coordinates.

        Returns:
            np.ndarray: A zero-copy view with shape (N, 3).
        """
        return self._data[:self._current_size, :3]
    
    @property
    def radii(self) -> np.ndarray:
        """Gets particle radii.

        Returns:
            np.ndarray: A zero-copy view with shape (N,).
        """
        return self._data[:self._current_size, 3]

    @property
    def masses(self) -> np.ndarray:
        """Gets particle masses.

        Returns:
            np.ndarray: A zero-copy view with shape (N,).
        """
        return self._data[:self._current_size, 4]
        
    @property
    def current_size(self) -> int:
        """Gets the current total number of particles."""
        return self._current_size
        
    @property
    def max_size(self) -> int:
        """Gets the maximum allowed number of particles."""
        return len(self._data)

    def add_particle(self, x: float, y: float, z: float, r: float, m: float) -> None:
        """Adds a new particle. O(1) complexity.

        Args:
            x (float): X coordinate
            y (float): Y coordinate
            z (float): Z coordinate
            r (float): Radius
            m (float): Mass

        Raises:
            RuntimeError: If cluster reaches maximum capacity.
        """
        if self._current_size >= len(self._data):
            raise RuntimeError("Aggregate capacity exceeded")
        i = self._current_size
        self._data[i] = [x, y, z, r, m]
        self._current_size += 1
        
    def to_numpy(self) -> np.ndarray:
        """Exports a copy of valid particles.

        Returns:
            np.ndarray: A copy of the data with shape (N, 5).
        """
        return self._data[:self._current_size].copy()
