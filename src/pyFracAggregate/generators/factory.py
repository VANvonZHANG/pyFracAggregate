"""Generator factory: legality matrix, aliases, generator construction."""
import warnings
from typing import Any

from pyFracAggregate.core.distributions import ParticleDistribution
from pyFracAggregate.core.scaling import ScalingLaw, get_scaling
from pyFracAggregate.generators.base import BaseGenerator
from pyFracAggregate.generators.cca import CCAGenerator
from pyFracAggregate.generators.pca import PCAGenerator

_GENERATORS = {
    "pca": PCAGenerator,
    "cca": CCAGenerator,
}

_VALID_PLACEMENTS = {
    "pca": ("sampled", "solved"),
    "cca": ("sampled", "solved", "constructed"),
}

_METHOD_ALIASES = {"fracval": ("cca", "mass", "constructed")}
_PLACEMENT_ALIASES = {"algebraic": "solved", "random": "sampled"}


def _resolve(method: Any, scaling: Any, placement: Any) -> "tuple[str, Any, str]":
    method = str(method).lower()
    if method in _METHOD_ALIASES:
        new_method, alias_scaling, alias_placement = _METHOD_ALIASES[method]
        warnings.warn(
            f"method='{method}' is deprecated since v0.4; use method='{new_method}', "
            f"scaling='{alias_scaling}', placement='{alias_placement}'. "
            "The alias will be removed in 1.0.",
            DeprecationWarning, stacklevel=3,
        )
        return new_method, alias_scaling, alias_placement

    if method == "tdcca":
        raise ValueError(
            "method='tdcca' was removed in v0.4 without a replacement "
            "(the implementation was not a faithful Thouy & Jullien 1994 "
            "port). Pin pyFracAggregate<0.4 if you need it."
        )
    if method in ("flage_pca", "flage_cca"):
        raise ValueError(
            f"method='{method}' has been removed. Use method='{'pca' if method == 'flage_pca' else 'cca'}' "
            "(FLAGE is the default placement strategy 'solved'). "
            "For the old random sampling behavior, use placement='sampled'."
        )
    if method not in _GENERATORS:
        raise ValueError(
            f"Unknown generation method: {method!r}. "
            f"Valid values: {sorted(_GENERATORS)}"
        )

    placement = (
        placement if isinstance(placement, str)
        else type(placement).__name__[:-len("Placement")].lower()
    )
    placement = str(placement).lower()
    if placement in _PLACEMENT_ALIASES:
        new_name = _PLACEMENT_ALIASES[placement]
        warnings.warn(
            f"placement='{placement}' is deprecated since v0.4; use placement='{new_name}'. "
            "The alias will be removed in 1.0.",
            DeprecationWarning, stacklevel=3,
        )
        placement = new_name
    if placement not in ("sampled", "solved", "constructed"):
        raise ValueError(
            f"Unknown placement strategy: {placement!r}. "
            "Valid values: 'sampled', 'solved', 'constructed'."
        )

    if placement not in _VALID_PLACEMENTS[method]:
        valid = ", ".join(repr(p) for p in _VALID_PLACEMENTS[method])
        raise ValueError(
            f"method='{method}' does not support placement='{placement}'. "
            f"Valid placements for '{method}': {valid}."
        )
    return method, scaling, placement


def get_generator(
    method: str,
    n_particles: int,
    df: float,
    kf: float,
    particle_dist: ParticleDistribution,
    overlap_tolerance: float = 1e-5,
    scaling: "str | ScalingLaw" = "mass",
    placement: "str | Any" = "solved",
    seed: "int | None" = None,
    **kwargs: Any,
) -> BaseGenerator:
    """Build a generator, validating the method x scaling x placement matrix.

    Args:
        method: 'pca' or 'cca' ('fracval' is a deprecated alias for
            (cca, mass, constructed); 'tdcca' was removed in v0.4).
        n_particles: Target number of primary particles.
        df: Fractal dimension.
        kf: Fractal prefactor.
        particle_dist: Primary radius distribution.
        overlap_tolerance: Allowed interpenetration between sphere surfaces.
        scaling: 'count', 'mass', or a ScalingLaw instance.
        placement: 'sampled', 'solved', 'constructed', or a
            PlacementStrategy instance.
        seed: Seed for reproducible generation (None = fresh entropy).
        **kwargs: Forwarded to the generator (e.g. surface_beta, density,
            length_unit, mass_unit).

    Raises:
        ValueError: Unknown method/scaling/placement or an illegal
            method x placement combination.
        TypeError: surface_beta with a placement other than 'solved'.
    """
    method, scaling, placement_name = _resolve(method, scaling, placement)

    surface_beta = kwargs.pop("surface_beta", None)
    if surface_beta is not None and placement_name != "solved":
        raise TypeError("surface_beta only applies to placement='solved'.")
    if surface_beta is not None:
        kwargs["surface_beta"] = surface_beta

    # Injected instances keep their configuration; the generator adopts them
    # (and shares its seeded Generator with them).
    from pyFracAggregate.generators.placement.base import PlacementStrategy
    placement_arg = (
        placement if isinstance(placement, PlacementStrategy) else placement_name
    )
    return _GENERATORS[method](
        n_particles, df, kf, particle_dist, overlap_tolerance,
        scaling=get_scaling(scaling, df, kf),
        placement=placement_arg, seed=seed, **kwargs,
    )
