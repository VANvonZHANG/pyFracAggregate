import numpy as np
from typing import List

from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.generators.base import BaseGenerator


# 26 lattice neighbor directions (face + edge + corner)
_LATTICE_DIRS: List[np.ndarray] = []
for _dx in [-1, 0, 1]:
    for _dy in [-1, 0, 1]:
        for _dz in [-1, 0, 1]:
            if _dx == 0 and _dy == 0 and _dz == 0:
                continue
            v = np.array([_dx, _dy, _dz], dtype=np.float64)
            _LATTICE_DIRS.append(v / np.linalg.norm(v))


class ThouyJullienGenerator(BaseGenerator):
    """Lattice-based tunable Cluster-Cluster Aggregation (Thouy & Jullien, 1994).

    Requires N = 2^n. Uses hierarchical pairing with Gamma deviation minimization.

    Key differences from Filippov CCA:
    - k^2 = 4 * (4^(1/Df) - 1) from Thouy & Jullien Eq 11
    - Target Gamma^2 = k^2 * (Rg1^2 + Rg2^2)/2 + 1 from Eq 12
    - The "+1" correction ensures exact results at the dimer stage
    - Max achievable Df ~ 2.55 in 3D due to geometric frustration
    """

    def generate(self) -> Aggregate:
        n = self.n_particles
        if n < 2 or (n & (n - 1)) != 0:
            raise ValueError(
                f"Thouy-Jullien tdCCA requires N to be a power of 2, got {n}"
            )

        agg = Aggregate(n, self.length_unit, self.mass_unit, self.density)
        radii = self.particle_dist.sample(n)
        masses = self.density * (4.0 / 3.0) * np.pi * (radii ** 3)
        a = np.mean(radii)

        # Each cluster: dict with positions, radii, masses arrays
        clusters: List[dict] = []
        for i in range(n):
            clusters.append({
                'positions': np.zeros((1, 3)),
                'radii': np.array([radii[i]]),
                'masses': np.array([masses[i]]),
            })

        # Iteration 1: form dimers using lattice neighbor directions
        new_clusters = []
        for i in range(0, n, 2):
            dimer = self._form_dimer(clusters[i], clusters[i + 1], a)
            new_clusters.append(dimer)
        clusters = new_clusters

        # Subsequent iterations: hierarchical merging
        size = 2
        while size < n:
            new_clusters = []
            for i in range(0, len(clusters), 2):
                merged = self._merge_min_gamma(clusters[i], clusters[i + 1], a)
                new_clusters.append(merged)
            clusters = new_clusters
            size *= 2

        # Populate the Aggregate from the final cluster
        final = clusters[0]
        for i in range(len(final['radii'])):
            pos = final['positions'][i]
            agg.add_particle(pos[0], pos[1], pos[2], final['radii'][i], final['masses'][i])
        return agg

    def _form_dimer(self, c1: dict, c2: dict, a: float) -> dict:
        """Form a dimer by placing two single particles adjacent in a random
        lattice direction."""
        direction = _LATTICE_DIRS[np.random.randint(len(_LATTICE_DIRS))]
        p1 = c1['positions'][0]
        p2 = p1 + direction * (c1['radii'][0] + c2['radii'][0])
        return {
            'positions': np.array([p1, p2]),
            'radii': np.concatenate([c1['radii'], c2['radii']]),
            'masses': np.concatenate([c1['masses'], c2['masses']]),
        }

    def _merge_min_gamma(self, c1: dict, c2: dict, a: float) -> dict:
        """Merge two clusters by finding a configuration that minimizes
        deviation from the target Gamma^2."""
        pos1 = c1['positions']
        pos2 = c2['positions']
        r1 = c1['radii']
        r2 = c2['radii']
        m1 = c1['masses']
        m2 = c2['masses']
        N1, N2 = len(r1), len(r2)

        # Center clusters at origin
        com1 = np.average(pos1, weights=m1, axis=0) if np.sum(m1) > 0 else np.mean(pos1, axis=0)
        com2 = np.average(pos2, weights=m2, axis=0) if np.sum(m2) > 0 else np.mean(pos2, axis=0)
        p1c = pos1 - com1
        p2c = pos2 - com2

        # Rg^2 of each cluster (mass-weighted, without intrinsic term)
        M1 = np.sum(m1)
        M2 = np.sum(m2)
        Rg1_sq = np.sum(m1[:, None] * (p1c ** 2)) / M1 if M1 > 0 else 0.0
        Rg2_sq = np.sum(m2[:, None] * (p2c ** 2)) / M2 if M2 > 0 else 0.0
        Rg_avg_sq = (Rg1_sq + Rg2_sq) / 2.0

        # k^2 from Thouy & Jullien Eq 11
        k_sq = 4.0 * (4.0 ** (1.0 / self.df) - 1.0)

        # Target Gamma^2 from Thouy & Jullien Eq 12
        # The "+1" correction ensures exact results at the dimer stage
        Gamma_target_sq = k_sq * Rg_avg_sq + 1.0

        # Find surface particles on each cluster
        surface1 = self._find_surface(p1c, r1, a)
        surface2 = self._find_surface(p2c, r2, a)

        if not surface1 or not surface2:
            return self._merge_fallback(c1, c2, a)

        # Try random orientations and positions, pick the one minimizing
        # |Gamma_actual^2 - Gamma_target^2|
        best_config = None
        best_delta = float('inf')

        max_attempts = 500

        for _ in range(max_attempts):
            # Random direction for cluster separation
            u = np.random.normal(size=3)
            u /= np.linalg.norm(u)

            # Scale the target Gamma to physical units (in units of a)
            Gamma_target = np.sqrt(max(Gamma_target_sq, 0.1)) * a

            # Place c2's center at approximate distance Gamma from origin
            # with some randomness to explore configurations
            scale = Gamma_target * np.random.uniform(0.7, 1.3)
            offset = u * scale
            p2_trial = p2c + offset

            # Fast overlap check with sampled particles
            ok = True
            sample1 = surface1 if len(surface1) < N1 else list(range(min(N1, 20)))
            sample2 = surface2 if len(surface2) < N2 else list(range(min(N2, 20)))
            for ii in sample1:
                for jj in sample2:
                    d = np.linalg.norm(p1c[ii] - p2_trial[jj])
                    if d < r1[ii] + r2[jj] - 1e-6:
                        ok = False
                        break
                if not ok:
                    break

            if not ok:
                continue

            # Full overlap check using broadcasting
            dists = np.linalg.norm(
                p1c[:, np.newaxis, :] - p2_trial[np.newaxis, :, :], axis=2
            )
            min_dists = r1[:, np.newaxis] + r2[np.newaxis, :]
            if np.any(dists < min_dists - 1e-6):
                continue

            # Compute Gamma: distance between the two cluster centers
            c2_center = (
                np.average(p2_trial, weights=m2, axis=0)
                if M2 > 0
                else np.mean(p2_trial, axis=0)
            )
            Gamma_actual_sq = np.sum(c2_center ** 2)
            delta = abs(Gamma_actual_sq - Gamma_target_sq * a * a)

            if delta < best_delta:
                best_delta = delta
                best_config = p2_trial.copy()

                # Early exit if very close to target
                if delta < 0.01 * a * a:
                    break

        if best_config is None:
            return self._merge_fallback(c1, c2, a)

        return {
            'positions': np.vstack([p1c, best_config]),
            'radii': np.concatenate([r1, r2]),
            'masses': np.concatenate([m1, m2]),
        }

    def _find_surface(
        self, positions: np.ndarray, radii: np.ndarray, a: float
    ) -> List[int]:
        """Find surface particles (those with at least one free face direction)."""
        n = len(radii)
        surface = []
        touch_dist = 2.0 * a

        for i in range(n):
            is_surface = False
            # Check 6 face neighbor directions
            for direction in _LATTICE_DIRS[:6]:
                neighbor_pos = positions[i] + direction * touch_dist
                has_neighbor = False
                for j in range(n):
                    if j == i:
                        continue
                    if np.linalg.norm(positions[j] - neighbor_pos) < radii[j] + 0.5 * a:
                        has_neighbor = True
                        break
                if not has_neighbor:
                    is_surface = True
                    break
            if is_surface:
                surface.append(i)
        return surface

    def _merge_fallback(self, c1: dict, c2: dict, a: float) -> dict:
        """Fallback: place c2 adjacent to c1 in a random lattice direction."""
        pos1 = c1['positions']
        pos2 = c2['positions']
        com1 = np.mean(pos1, axis=0)
        com2 = np.mean(pos2, axis=0)

        p1c = pos1 - com1
        p2c = pos2 - com2

        direction = _LATTICE_DIRS[np.random.randint(len(_LATTICE_DIRS))]
        max_extent = (
            np.max(np.linalg.norm(p1c, axis=1) + c1['radii'])
            + np.max(np.linalg.norm(p2c, axis=1) + c2['radii'])
        )
        offset = direction * (max_extent + a)
        p2_final = p2c + offset

        return {
            'positions': np.vstack([p1c, p2_final]),
            'radii': np.concatenate([c1['radii'], c2['radii']]),
            'masses': np.concatenate([c1['masses'], c2['masses']]),
        }
