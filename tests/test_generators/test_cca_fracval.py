import pytest
import numpy as np

import pyFracAggregate as pfa

def test_cca_fracval_generation():
    # 测试 FracVAL 方法的多分散生成
    dist = pfa.LognormalDistribution(mean=1.0, std=1.2)
    agg = pfa.generate(
        n_particles=15,
        df=1.8,
        kf=1.3,
        method='fracval',
        particle_dist=dist
    )
    
    assert agg.current_size == 15
    
    positions = agg.positions
    radii = agg.radii
    overlap_tolerance = 1e-4
    
    # 检查是否有严重的重叠
    for i in range(15):
        for j in range(i + 1, 15):
            dist_val = np.linalg.norm(positions[i] - positions[j])
            min_dist = radii[i] + radii[j] - overlap_tolerance
            assert dist_val >= min_dist - 1e-3

def test_cca_fracval_small_particles():
    # 对于小粒子数 (<=8)，算法会自动退化到 PCA
    agg = pfa.generate(
        n_particles=5,
        df=1.8,
        kf=1.3,
        method='fracval'
    )
    assert agg.current_size == 5
