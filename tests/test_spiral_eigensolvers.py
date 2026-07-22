import math
import unittest

import torch

from src.klap_eigenfunctions import KlapConfig, fit_klap_eigenfunctions
from src.spin_eigenfunctions import SpINConfig, fit_spin_eigenfunctions


class ReducedSpiralIntegrationTests(unittest.TestCase):
    def test_both_methods_build_psd_three_mode_heat_kernels(self):
        generator = torch.Generator().manual_seed(12)

        def spiral_sample(count):
            t = 4.0 * math.pi * torch.rand(count, generator=generator)
            radius = 1.2 - 0.7 * t / (4.0 * math.pi)
            points = torch.stack((radius * torch.cos(t), radius * torch.sin(t)), dim=1)
            return points + 0.025 * torch.randn(
                count, 2, generator=generator
            )

        reference = spiral_sample(72)
        validation = spiral_sample(68)
        evaluation = spiral_sample(31)
        reference_weights = torch.ones(len(reference))
        validation_weights = torch.ones(len(validation))

        klap = fit_klap_eigenfunctions(
            reference.double(),
            reference_weights.double(),
            KlapConfig(
                num_nonconstant_modes=2,
                num_centers=14,
                bandwidth_candidates=(0.3, 0.5),
                seed=12,
            ),
            validation_points=validation.double(),
            validation_probability_weights=validation_weights.double(),
            verbose=False,
        )
        spin = fit_spin_eigenfunctions(
            reference,
            reference_weights,
            SpINConfig(
                num_nonconstant_modes=2,
                hidden_width=12,
                hidden_layers=1,
                fourier_frequencies=1,
                steps=8,
                batch_size=24,
                seed=12,
                log_every=8,
            ),
            validation_points=validation,
            validation_probability_weights=validation_weights,
            verbose=False,
        )

        for fit, query in ((klap, evaluation.double()), (spin, evaluation)):
            with torch.no_grad():
                values = fit.system(query)
            self.assertEqual(values.shape, (len(evaluation), 3))
            spectral_weights = torch.exp(-fit.eigenvalues)
            kernel = (values * spectral_weights.sqrt()) @ (
                values * spectral_weights.sqrt()
            ).T
            self.assertTrue(torch.isfinite(kernel).all())
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(kernel).min()), -2e-5)


if __name__ == "__main__":
    unittest.main()
