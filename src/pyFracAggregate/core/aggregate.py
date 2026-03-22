import numpy as np

class Aggregate:
    """
    核心物理实体，表示一个分形团簇。
    基于 NumPy 预分配连续内存矩阵，提供极高的数据局部性和访问性能。
    """
    def __init__(self, max_particles: int):
        """
        初始化 Aggregate 对象。

        Args:
            max_particles (int): 团簇预计能容纳的最大粒子数。

        Raises:
            ValueError: 如果 max_particles <= 0。
        """
        if max_particles <= 0:
            raise ValueError("max_particles must be positive")
        
        # 数据结构为 [x, y, z, radius, mass] 预分配内存
        self._data = np.zeros((max_particles, 5), dtype=np.float64)
        self._current_size = 0  

    @property
    def positions(self) -> np.ndarray:
        """
        获取粒子的坐标。

        Returns:
            np.ndarray: 形状为 (N, 3) 的零拷贝视图 (Zero-copy view)。
        """
        return self._data[:self._current_size, :3]
    
    @property
    def radii(self) -> np.ndarray:
        """
        获取粒子的半径。

        Returns:
            np.ndarray: 形状为 (N,) 的零拷贝视图。
        """
        return self._data[:self._current_size, 3]

    @property
    def masses(self) -> np.ndarray:
        """
        获取粒子的质量。

        Returns:
            np.ndarray: 形状为 (N,) 的零拷贝视图。
        """
        return self._data[:self._current_size, 4]
        
    @property
    def current_size(self) -> int:
        """获取当前包含的粒子总数。"""
        return self._current_size
        
    @property
    def max_size(self) -> int:
        """获取最大允许的粒子数。"""
        return len(self._data)

    def add_particle(self, x: float, y: float, z: float, r: float, m: float) -> None:
        """
        添加一个新粒子。O(1) 复杂度。

        Args:
            x (float): X 坐标
            y (float): Y 坐标
            z (float): Z 坐标
            r (float): 半径
            m (float): 质量

        Raises:
            RuntimeError: 如果团簇达到最大容量导致越界。
        """
        if self._current_size >= len(self._data):
            raise RuntimeError("Aggregate capacity exceeded")
        i = self._current_size
        self._data[i] = [x, y, z, r, m]
        self._current_size += 1
        
    def to_numpy(self) -> np.ndarray:
        """
        导出当前有效粒子的拷贝。

        Returns:
            np.ndarray: 形状为 (N, 5) 的数据拷贝。
        """
        return self._data[:self._current_size].copy()
