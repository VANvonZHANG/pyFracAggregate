import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.generators.base import BaseGenerator
from pyFracAggregate.generators.pca import PCAGenerator
from pyFracAggregate.analysis.morphology import center_of_mass, radius_of_gyration


class CCAGenerator(BaseGenerator):
    """Cluster-Cluster Aggregation with pluggable placement strategy."""

    def generate(self) -> Aggregate:
        if self.n_particles <= 8:
            pca_gen = PCAGenerator(
                self.n_particles, self.df, self.kf, self.particle_dist,
                self.overlap_tolerance, self.length_unit, self.mass_unit,
                self.density, placement=self._placement_name
            )
            return pca_gen.generate()

        radii = self.particle_dist.sample(self.n_particles)

        cluster_size = 5
        cluster_list = []
        idx = 0
        while idx < self.n_particles:
            rem = self.n_particles - idx
            curr_size = cluster_size if rem >= cluster_size * 1.5 else rem

            class LocalDist:
                def __init__(self, r):
                    self.r = r
                def sample(self, n):
                    return self.r

            local_pca = PCAGenerator(
                curr_size, self.df, self.kf,
                LocalDist(radii[idx:idx + curr_size]),
                self.overlap_tolerance, self.length_unit, self.mass_unit,
                self.density, placement=self._placement_name
            )
            sub_agg = local_pca.generate()
            cluster_list.append(sub_agg)
            idx += curr_size

        while len(cluster_list) > 1:
            agg1 = cluster_list.pop(0)
            agg2 = cluster_list.pop(0)
            merged = self._merge(agg1, agg2)
            cluster_list.append(merged)

        return cluster_list[0]

    def _merge(self, agg1: Aggregate, agg2: Aggregate) -> Aggregate:
        N1 = agg1.current_size
        N2 = agg2.current_size
        N = N1 + N2

        com1 = center_of_mass(agg1)
        com2 = center_of_mass(agg2)
        Rg1 = radius_of_gyration(agg1)
        Rg2 = radius_of_gyration(agg2)

        a = (np.mean(agg1.radii) * N1 + np.mean(agg2.radii) * N2) / N

        term1 = (a**2 * N**2) / (N1 * N2) * (N / self.kf) ** (2.0 / self.df)
        term2 = (N / N2) * Rg1**2
        term3 = (N / N1) * Rg2**2

        Gamma_sq = term1 - term2 - term3
        Gamma = np.sqrt(max(Gamma_sq, 0.0))

        pos1 = agg1.positions - com1
        r1 = agg1.radii
        pos2_centered = agg2.positions - com2
        r2 = agg2.radii

        pos2_final = self.placement.merge_clusters(
            pos1, r1, agg1, pos2_centered, r2, agg2, Gamma, a
        )

        if pos2_final is None:
            pos2_final = pos2_centered

        merged = Aggregate(N, self.length_unit, self.mass_unit, self.density)
        for i in range(N1):
            merged.add_particle(
                pos1[i, 0], pos1[i, 1], pos1[i, 2],
                agg1.radii[i], agg1.masses[i]
            )
        for j in range(N2):
            merged.add_particle(
                pos2_final[j, 0], pos2_final[j, 1], pos2_final[j, 2],
                agg2.radii[j], agg2.masses[j]
            )
        return merged
