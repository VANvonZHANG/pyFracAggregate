import numpy as np
from typing import List, Optional
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.math_utils import (
    rotate_points,
    euler_rodrigues_rotation,
    sphere_sphere_intersection,
    random_point_on_circle,
)
from pyFracAggregate.generators.base import BaseGenerator
from pyFracAggregate.generators.pca_flage import PCAFlageGenerator
from pyFracAggregate.analysis.morphology import center_of_mass, radius_of_gyration


class FracVALGenerator(BaseGenerator):
    """FracVAL polydisperse algorithm (Moran et al., 2019).

    Generates precise fractal structures by combining polydisperse mass weights
    and deterministic contact placement with hierarchical merging.
    """

    def generate(self) -> Aggregate:
        if self.n_particles <= 8:
            pca_gen = PCAFlageGenerator(
                self.n_particles, self.df, self.kf, self.particle_dist,
                self.overlap_tolerance, self.length_unit, self.mass_unit, self.density
            )
            return pca_gen.generate()

        radii = self.particle_dist.sample(self.n_particles)

        if self.n_particles < 50:
            cluster_size = 5
        elif self.n_particles <= 500:
            cluster_size = max(5, int(self.n_particles * 0.1))
        else:
            cluster_size = 50

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

            local_pca = PCAFlageGenerator(
                curr_size, self.df, self.kf,
                LocalDist(radii[idx:idx + curr_size]),
                self.overlap_tolerance, self.length_unit, self.mass_unit, self.density
            )
            sub_agg = local_pca.generate()
            cluster_list.append(sub_agg)
            idx += curr_size

        while len(cluster_list) > 1:
            agg1 = cluster_list.pop(0)
            agg2 = cluster_list.pop(0)
            merged_agg = self._merge_fracval(agg1, agg2)
            cluster_list.append(merged_agg)

        return cluster_list[0]

    def _merge_fracval(self, agg1: Aggregate, agg2: Aggregate) -> Aggregate:
        N1 = agg1.current_size
        N2 = agg2.current_size
        N = N1 + N2

        com1 = center_of_mass(agg1)
        com2 = center_of_mass(agg2)
        Rg1 = radius_of_gyration(agg1)
        Rg2 = radius_of_gyration(agg2)

        m1 = np.sum(agg1.masses)
        m2 = np.sum(agg2.masses)
        m = m1 + m2

        r_p_geo = np.mean(np.concatenate([agg1.radii, agg2.radii]))

        # Moran 2019 Eq 3 & 6: mass-weighted Gamma calculation
        Rg = r_p_geo * (N / self.kf) ** (1.0 / self.df)
        term_target = m**2 * Rg**2
        term_parts = m * (m1 * Rg1**2 + m2 * Rg2**2)
        Gamma_sq = (term_target - term_parts) / (m1 * m2)
        Gamma = np.sqrt(max(Gamma_sq, 0.0))

        pos1 = agg1.positions - com1
        pos2_centered = agg2.positions - com2
        r1 = agg1.radii
        r2 = agg2.radii

        # Sub-step b: Build binary contact matrix
        D_i_plus = np.linalg.norm(pos1, axis=1) + r1
        D_j_plus = np.linalg.norm(pos2_centered, axis=1) + r2
        contact_threshold = Gamma - 1e-10
        contact_mask = (D_i_plus[:, np.newaxis] + D_j_plus[np.newaxis, :]) >= contact_threshold
        contact_pairs = list(zip(*np.where(contact_mask)))

        if len(contact_pairs) == 0:
            return self._merge_random_fallback(
                pos1, pos2_centered, r1, r2, agg1, agg2, Gamma, r_p_geo, N
            )

        np.random.shuffle(contact_pairs)
        max_pair_attempts = min(len(contact_pairs), 50)

        for pair_idx in range(max_pair_attempts):
            si_idx, sj_idx = contact_pairs[pair_idx]
            si_pos = pos1[si_idx]
            sj_pos = pos2_centered[sj_idx]

            # Sub-step c Stage 1: determine contact geometry
            si_direction = si_pos / max(np.linalg.norm(si_pos), 1e-12)
            contact_point_si = si_pos + r1[si_idx] * si_direction

            # Place CM2 so sj's surface can reach contact_point_si
            cp_result = sphere_sphere_intersection(
                np.zeros(3), Gamma, contact_point_si, r2[sj_idx]
            )

            if cp_result is not None:
                cc, cr = cp_result
                if cr > 1e-10:
                    cm2_pos = random_point_on_circle(cc, cr, si_direction)
                else:
                    cm2_pos = cc
            else:
                # Fallback: random CM2 placement on Gamma sphere
                u = np.random.normal(size=3)
                u /= np.linalg.norm(u)
                cm2_pos = Gamma * u

            # Stage 2: Rotate A2 so sj aligns toward contact point
            sj_target = contact_point_si - r2[sj_idx] * si_direction
            v_sj = sj_target - cm2_pos
            v_sj_len = np.linalg.norm(v_sj)
            if v_sj_len < 1e-12:
                continue

            sj_desired_dir = v_sj / v_sj_len
            sj_current_dir = sj_pos / max(np.linalg.norm(sj_pos), 1e-12)

            rot_axis = np.cross(sj_current_dir, sj_desired_dir)
            rot_axis_len = np.linalg.norm(rot_axis)
            if rot_axis_len < 1e-12:
                angle = 0.0
            else:
                rot_axis = rot_axis / rot_axis_len
                cos_angle = np.clip(np.dot(sj_current_dir, sj_desired_dir), -1.0, 1.0)
                angle = np.arccos(cos_angle)

            pos2_aligned = euler_rodrigues_rotation(pos2_centered, rot_axis, angle)

            # Translate A2 so sj's center goes to sj_target
            sj_current = pos2_aligned[sj_idx]
            translation = sj_target - sj_current
            pos2_final = pos2_aligned + translation

            # Check overlaps
            dists = np.linalg.norm(
                pos1[:, np.newaxis, :] - pos2_final[np.newaxis, :, :], axis=2
            )
            min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - self.overlap_tolerance
            gaps = dists - min_dists

            if np.any(gaps < -1e-10):
                # Sub-step d: rotate A2 around contact axis to resolve overlaps
                contact_axis = si_direction
                resolved = False
                for _ in range(25):
                    rand_angle = np.random.uniform(0, 2 * np.pi)
                    pos2_rotated = euler_rodrigues_rotation(
                        pos2_aligned, contact_axis, rand_angle
                    )
                    sj_curr = pos2_rotated[sj_idx]
                    trans = sj_target - sj_curr
                    pos2_final = pos2_rotated + trans

                    dists = np.linalg.norm(
                        pos1[:, np.newaxis, :] - pos2_final[np.newaxis, :, :], axis=2
                    )
                    gaps = dists - min_dists
                    if not np.any(gaps < -1e-10):
                        resolved = True
                        break
                if not resolved:
                    continue

            return self._build_merged(pos1, agg1, pos2_final, agg2, N)

        # All pairs exhausted, fall back to random
        return self._merge_random_fallback(
            pos1, pos2_centered, r1, r2, agg1, agg2, Gamma, r_p_geo, N
        )

    def _merge_random_fallback(
        self, pos1, pos2_centered, r1, r2, agg1, agg2, Gamma, a, N
    ):
        max_attempts = 50000
        tolerance = 1e-3 * a
        best_candidate = None
        min_gap = float('inf')
        candidate_pos2 = None

        for attempt in range(max_attempts):
            u = np.random.normal(size=3)
            u /= np.linalg.norm(u)
            new_com2 = Gamma * u

            euler_angles = np.random.uniform(0, 2 * np.pi, size=3)
            pos2_rotated = rotate_points(pos2_centered, tuple(euler_angles))
            candidate_pos2 = pos2_rotated + new_com2

            dists = np.linalg.norm(
                pos1[:, np.newaxis, :] - candidate_pos2[np.newaxis, :, :], axis=2
            )
            min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - self.overlap_tolerance
            gaps = dists - min_dists

            if np.any(gaps < 0):
                continue
            current_min_gap = np.min(gaps)
            if current_min_gap < min_gap:
                min_gap = current_min_gap
                best_candidate = candidate_pos2.copy()
            if current_min_gap <= tolerance:
                return self._build_merged(pos1, agg1, candidate_pos2, agg2, N)
            if attempt > 0 and attempt % 2000 == 0:
                tolerance += 0.05 * a

        if best_candidate is None:
            best_candidate = candidate_pos2
        return self._build_merged(pos1, agg1, best_candidate, agg2, N)

    def _build_merged(self, pos1, agg1, pos2_final, agg2, N):
        merged = Aggregate(N, self.length_unit, self.mass_unit, self.density)
        for i in range(agg1.current_size):
            merged.add_particle(
                pos1[i, 0], pos1[i, 1], pos1[i, 2],
                agg1.radii[i], agg1.masses[i]
            )
        for j in range(agg2.current_size):
            merged.add_particle(
                pos2_final[j, 0], pos2_final[j, 1], pos2_final[j, 2],
                agg2.radii[j], agg2.masses[j]
            )
        return merged
