import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.math_utils import (
    euler_rodrigues_rotation,
    random_point_on_circle,
    sphere_sphere_intersection,
)
from pyFracAggregate.generators.placement.base import PlacementStrategy
from pyFracAggregate.generators.placement.solvers import mc_touch_merge


class ConstructedPlacement(PlacementStrategy):
    """Specified contact pair + attitude construction + COM correction
    (Moran et al., 2019, FracVAL sub-steps b-d)."""

    def __init__(self, overlap_tolerance: float = 1e-5,
                 rng: "np.random.Generator | None" = None):
        self.overlap_tolerance = overlap_tolerance
        self.rng = rng if rng is not None else np.random.default_rng()

    def place_particle(
        self,
        agg: Aggregate,
        candidate_radius: float,
        candidate_mass: float,
        geom_center: np.ndarray,
        L: float,
        mean_radius: float,
    ) -> tuple | None:
        raise NotImplementedError(
            "constructed placement applies to cluster merging "
            "(method='cca') only; single particles cannot be 'constructed'."
        )

    def merge_clusters(
        self,
        pos1: np.ndarray,
        r1: np.ndarray,
        agg1: Aggregate,
        pos2_centered: np.ndarray,
        r2: np.ndarray,
        agg2: Aggregate,
        Gamma: float,
        mean_radius: float,
    ) -> np.ndarray:
        # Sub-step b: build binary contact matrix
        D_i_plus = np.linalg.norm(pos1, axis=1) + r1
        D_j_plus = np.linalg.norm(pos2_centered, axis=1) + r2
        contact_threshold = Gamma - 1e-10
        contact_mask = (D_i_plus[:, np.newaxis] + D_j_plus[np.newaxis, :]) >= contact_threshold
        contact_pairs = list(zip(*np.where(contact_mask)))

        if len(contact_pairs) == 0:
            return mc_touch_merge(pos1, r1, pos2_centered, r2, Gamma, mean_radius,
                                  self.overlap_tolerance, self.rng,
                                  track_best=True, max_attempts=50000)

        self.rng.shuffle(contact_pairs)
        max_pair_attempts = min(len(contact_pairs), 50)

        for pair_idx in range(max_pair_attempts):
            si_idx, sj_idx = contact_pairs[pair_idx]
            si_pos = pos1[si_idx]
            sj_pos = pos2_centered[sj_idx]

            # Sub-step c stage 1: contact geometry
            si_direction = si_pos / max(float(np.linalg.norm(si_pos)), 1e-12)
            contact_point_si = si_pos + r1[si_idx] * si_direction

            cp_result = sphere_sphere_intersection(
                np.zeros(3), Gamma, contact_point_si, r2[sj_idx]
            )
            if cp_result is not None:
                cc, cr = cp_result
                if cr > 1e-10:
                    cm2_pos = random_point_on_circle(cc, cr, si_direction, rng=self.rng)
                else:
                    cm2_pos = cc
            else:
                u = self.rng.normal(size=3)
                u /= np.linalg.norm(u)
                cm2_pos = Gamma * u

            # Stage 2: rotate A2 so sj aligns toward the contact point
            sj_target = contact_point_si - r2[sj_idx] * si_direction
            v_sj = sj_target - cm2_pos
            v_sj_len = np.linalg.norm(v_sj)
            if v_sj_len < 1e-12:
                continue

            sj_desired_dir = v_sj / v_sj_len
            sj_current_dir = sj_pos / max(float(np.linalg.norm(sj_pos)), 1e-12)

            rot_axis = np.cross(sj_current_dir, sj_desired_dir)
            rot_axis_len = np.linalg.norm(rot_axis)
            if rot_axis_len < 1e-12:
                angle = 0.0
            else:
                rot_axis = rot_axis / rot_axis_len
                cos_angle = np.clip(np.dot(sj_current_dir, sj_desired_dir), -1.0, 1.0)
                angle = np.arccos(cos_angle)

            pos2_aligned = euler_rodrigues_rotation(pos2_centered, rot_axis, angle)

            sj_current = pos2_aligned[sj_idx]
            translation = sj_target - sj_current
            pos2_final = pos2_aligned + translation

            dists = np.linalg.norm(
                pos1[:, np.newaxis, :] - pos2_final[np.newaxis, :, :], axis=2
            )
            min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - self.overlap_tolerance
            gaps = dists - min_dists

            if np.any(gaps < -1e-10):
                # Sub-step d: spin A2 around the contact axis to resolve overlaps
                contact_axis = si_direction
                resolved = False
                for _ in range(25):
                    rand_angle = self.rng.uniform(0, 2 * np.pi)
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

            # Verify CM2 constraint
            m2_total = np.sum(agg2.masses)
            actual_com2 = (np.average(pos2_final, weights=agg2.masses, axis=0)
                           if m2_total > 0 else np.mean(pos2_final, axis=0))
            com2_error = abs(np.linalg.norm(actual_com2) - Gamma)
            if com2_error > 0.1 * Gamma:
                continue

            return pos2_final

        return mc_touch_merge(pos1, r1, pos2_centered, r2, Gamma, mean_radius,
                              self.overlap_tolerance, self.rng,
                              track_best=True, max_attempts=50000)
