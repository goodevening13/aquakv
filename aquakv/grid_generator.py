import math
import torch
from tqdm.auto import trange
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser("Grid Generator")
    parser.add_argument("-d", "--dim", type=int)
    parser.add_argument("-n", "--n_points", type=int)
    parser.add_argument("-g", "--gamma_0", type=float, default=2.0, required=False)
    parser.add_argument(
        "-o", "--out_dir", type=Path, default=Path("grids"), required=False
    )
    parser.add_argument(
        "-N", "--n_samples", type=int, default=10_000_000, required=False
    )
    parser.add_argument("-s", "--steps", type=int, default=600, required=False)
    return parser.parse_args()


def sample_gaussian(n_samples: int, dim: int):
    return torch.empty(n_samples, dim, device="cuda").normal_(0, 1)


def compute_centers_of_mass_with_gaussian_weights(grid, sampled_points):
    """
    Compute centers of mass for Voronoi cells using points weighted by a Gaussian distribution, in a vectorized manner.
    """
    # Find the Voronoi cell each point belongs to
    closest_indices = torch.empty(
        sampled_points.shape[0], dtype=torch.long, device=sampled_points.device
    )
    for i in range(0, sampled_points.shape[0], 4096):
        closest_indices[i : i + 4096] = torch.argmax(
            2 * sampled_points[i : i + 4096] @ grid.T - torch.norm(grid, dim=1) ** 2,
            dim=1,
        )

    # Calculate variance
    variance = (sampled_points - grid[closest_indices]).pow(2).mean()

    # Initialize sums and weights sum arrays
    centers_of_mass = torch.zeros_like(grid)

    # Accumulate weighted points and total weights for each Voronoi cell
    for i in range(grid.shape[0]):
        cell_mask = closest_indices == i
        centers_of_mass[i] = torch.mean(sampled_points[cell_mask], dim=0)

    return centers_of_mass, variance


def get_gamma(n, GAMMA_0, A, B):
    return GAMMA_0 * A / (A + GAMMA_0 * B * n)


# Example usage
@torch.inference_mode()
def get_grid(dim, n_points, steps, n_samples, GAMMA_0, A, B):
    grid = torch.empty(n_points, dim, device="cuda").normal_(0, 1)

    sliding_variance = None

    for n in (pbar := trange(steps, leave=False)):
        sampled_points = sample_gaussian(n_samples=n_samples, dim=dim)

        # Compute centers of mass
        centers_of_mass, variance = compute_centers_of_mass_with_gaussian_weights(
            grid, sampled_points
        )
        gamma = get_gamma(n, GAMMA_0, A, B)
        grid = grid + gamma * (centers_of_mass - grid)

        if sliding_variance is None:
            sliding_variance = variance
        else:
            sliding_variance = 0.9 * sliding_variance + 0.1 * variance

        if n % 10 == 0:
            pbar.set_description(f"sliding variance = {sliding_variance.item()}")

    return centers_of_mass


if __name__ == "__main__":
    args = parse_args()

    A = 4 * args.n_points ** (1 / args.dim)
    B = math.pi**2 * args.n_points ** (-2 / args.dim)
    args = parse_args()

    grid = get_grid(
        args.dim, args.n_points, args.steps, args.n_samples, args.gamma_0, A, B
    )

    torch.save(torch.asarray(grid), args.out_dir / f"EDEN{args.dim}-{args.n_points}.pt")
