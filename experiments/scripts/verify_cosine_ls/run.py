"""Numerically verify the DCT-II Neumann cosine-domain LS implementation.

This is a diagnostic only; it does not use formal train/validation/test arrays.
"""

from __future__ import annotations

import numpy as np
from scipy import fft

from experiments.common.baselines import cosine_ls as solver


def main() -> None:
    rng = np.random.default_rng(20260910)
    alpha = 0.5
    lambda_ls = 0.3

    # 1) Spatial Neumann finite-difference operator must be diagonalized by DCT-II.
    for shape in ((4, 5), (7, 6), (11, 13)):
        image = rng.normal(size=shape)
        spatial = solver._apply_screened_operator(image, alpha)
        transformed = fft.dctn(spatial, type=2, norm="ortho")
        expected = solver._screened_operator_symbol(shape, alpha) * fft.dctn(
            image, type=2, norm="ortho"
        )
        error = float(np.max(np.abs(transformed - expected)))
        if error > 1e-9:
            raise AssertionError(f"DCT-II operator diagonalization failed: {shape}, {error}")

    # 2) Closed-form cosine-domain solution must equal the direct normal-equation solve.
    shape = (5, 6)
    image = rng.normal(size=shape)
    forcing = rng.normal(size=shape)
    symbol = solver._screened_operator_symbol(shape, alpha)
    image_coeff = fft.dctn(image, type=2, norm="ortho")
    forcing_coeff = fft.dctn(forcing, type=2, norm="ortho")
    background_coeff = (image_coeff - lambda_ls * symbol * forcing_coeff) / (
        1.0 + lambda_ls * symbol**2
    )
    cosine_solution = fft.idctn(background_coeff, type=2, norm="ortho")

    count = shape[0] * shape[1]
    basis = np.eye(count, dtype=np.float64)
    operator = np.column_stack(
        [
            solver._apply_screened_operator(basis[:, column].reshape(shape), alpha).reshape(-1)
            for column in range(count)
        ]
    )
    system = np.eye(count) + lambda_ls * operator.T @ operator
    rhs = image.reshape(-1) - lambda_ls * operator.T @ forcing.reshape(-1)
    direct_solution = np.linalg.solve(system, rhs).reshape(shape)
    error = float(np.max(np.abs(cosine_solution - direct_solution)))
    if error > 1e-9:
        raise AssertionError(f"closed-form/direct solve mismatch: {error}")

    # 3) Torch DCT helper (used by CUDA backend) must match SciPy's orthonormal DCT-II.
    if solver.torch is not None:
        tensor = solver.torch.as_tensor(rng.normal(size=(8, 9)), dtype=solver.torch.float64)
        torch_coeff = solver._dct2_torch(tensor).detach().cpu().numpy()
        scipy_coeff = fft.dctn(tensor.detach().cpu().numpy(), type=2, norm="ortho")
        error = float(np.max(np.abs(torch_coeff - scipy_coeff)))
        if error > 1e-9:
            raise AssertionError(f"torch/scipy DCT-II mismatch: {error}")
        roundtrip = solver._idct2_torch(solver._dct2_torch(tensor))
        error = float(
            np.max(np.abs(roundtrip.detach().cpu().numpy() - tensor.detach().cpu().numpy()))
        )
        if error > 1e-9:
            raise AssertionError(f"torch DCT-II roundtrip mismatch: {error}")

    print("Cosine-domain LS verification: PASS")
    print(" boundary = zero-normal-derivative (Neumann)")
    print(" grid = pixel-centred")
    print(" transform = orthonormal DCT-II")
    print(" closed-form solution = direct normal-equation solution")


if __name__ == "__main__":
    main()
