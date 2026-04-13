# GreenFunction Test Physics

This folder's tests are meant to check physical expectations, not implementation details. The statements below summarize what the test cases assume should be true in each scenario.

## Free-Space Expectations

The free-space tests treat the scalar Green's function as

$$
G(R) = \frac{e^{-jkR}}{R}
$$

with $R$ equal to the full 3D separation between source and observation points.

The physical expectations exercised by the tests are:

- The vector-potential and scalar-potential kernels are identical in homogeneous free space.
- The kernel depends only on separation distance, not on direction or coordinate orientation.
- Swapping source and observer does not change the result, i.e. reciprocity holds.
- Point-based evaluation and precomputed-distance evaluation must represent the same physics.
- The field magnitude decays like $1/R$, so it becomes small at very large distance.
- The same formula should remain valid for non-axis-aligned 3D geometry and for different floating-point precisions within tolerance.

## Perfect Electric Conductor Ground-Plane Expectations

The ground-plane tests model a source above an infinite PEC plane through image theory. The test suite currently expects the mirrored source point to be

$$
\mathbf{r}'_{\mathrm{img}} = (x', y', 2 z_{\mathrm{gnd}} - z')
$$

and expects both tested Green-function channels to use the same image subtraction form:

$$
G_{\mathrm{tested}} = G_{\mathrm{direct}} - G_{\mathrm{image}}
$$

where

$$
G_{\mathrm{direct}} = \frac{e^{-jkR_{\mathrm{direct}}}}{R_{\mathrm{direct}}},
\qquad
G_{\mathrm{image}} = \frac{e^{-jkR_{\mathrm{image}}}}{R_{\mathrm{image}}}
$$

The physical expectations encoded by the tests are:

- Mirroring across the ground plane changes only the $z$ coordinate.
- The image contribution must use the distance from the observer to the mirrored source.
- For the cases covered here, both tested channels are expected to subtract the image term and therefore be equal.
- At the PEC surface, the direct and image distances are equal, so cancellation occurs and the tested response vanishes there.
- A same-height formulation should agree with the general 3D formulation when source and observer are at the same elevation.
- Reciprocity is still expected when both points lie above the ground plane.
- Height-above-ground helper behavior is interpreted geometrically: a point on the plane has zero height and a point above it has positive height.
- Exposing direct and image components separately should reconstruct the total tested response through direct-minus-image.

## Layered-Stack Expectations

The layer-stack tests focus on simple physical geometry and material bookkeeping rather than full-wave field solutions.

The test expectations are:

- A layer is defined by name, relative permittivity, relative permeability, and thickness.
- A layer with infinite thickness is treated as a half-space.
- Interfaces accumulate from the reference plane upward according to the listed thicknesses.
- A point's layer index is determined only by its $z$ position relative to those interfaces.
- Two points are in the same layer only when both fall within the same interface bounds.
- Thickness-above and thickness-below queries are geometric distances to the nearest layer boundaries.
- In lossy media, effective permittivity is expected to gain a negative imaginary part.

## Configuration-Level Expectations

The configuration tests express physical intent at a higher level:

- Free space is the default propagation environment.
- Selecting a ground plane at a given $z$ should produce a model tied to that plane location.
- Explicit factory requests should let a test ask for a particular environment directly.
- Layered media is treated as a declared but not yet usable physical mode in these tests.
- An end-to-end ground-plane configuration should produce the same equality between the two tested channels that the ground-plane unit tests expect.
- An end-to-end free-space configuration should also produce equality between the two tested channels.

## Scope Note

This README intentionally describes only the physical behavior asserted by the tests in this folder. It does not explain or justify the production implementation outside these tests.