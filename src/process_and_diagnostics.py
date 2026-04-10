from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import os
import numpy as np
from causaltensor.matlib import SVD
import matplotlib.pyplot as plt

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'outputs')
PROCESS_AND_DIAGNOSTICS_DIR = os.path.join(OUTPUT_DIR, 'process_and_diagnostics')

class ProcessAndDiagnostics:
    """Run imputation and SVD diagnostics for real/synthetic matrices with NaNs.
    
    This class provides tools for matrix completion using hard impute SVD and comprehensive
    SVD diagnostics to compare real and synthetic data matrices.

    Attributes:
        real (np.ndarray): Input real-valued matrix with potential NaNs.
        synthetic (np.ndarray): Input synthetic-valued matrix with potential NaNs.
        real_col_means (np.ndarray): Column means from real matrix for normalization.
        real_col_stds (np.ndarray): Column standard deviations from real matrix for normalization.
        real_normalized (np.ndarray): Normalized real matrix.
        synthetic_normalized (np.ndarray): Normalized synthetic matrix.
        real_diagnostics (Optional[Dict]): SVD diagnostics for real matrix.
        synthetic_diagnostics (Optional[Dict]): SVD diagnostics for synthetic matrix.
        stacked_diagnostics (Optional[Dict]): SVD diagnostics for stacked matrices.
        real_completed (Optional[np.ndarray]): Completed real matrix after imputation.
        synthetic_completed (Optional[np.ndarray]): Completed synthetic matrix after imputation.
    """

    def __init__(
        self,
        real_matrix: np.ndarray,
        synthetic_matrix: np.ndarray,
        dataset_name: str
    ) -> None:
        """Initialize with real and synthetic matrices, normalize them, and prepare for diagnostics.

        Args:
            real_matrix: Real-valued matrix of shape (n_rows, n_cols), may contain NaNs.
            synthetic_matrix: Synthetic-valued matrix of the same shape, may contain NaNs.
            dataset_name: Name of the dataset. Used for saving the results.

        Returns:
            None
        """
        if real_matrix.shape != synthetic_matrix.shape:
            raise ValueError("real_matrix and synthetic_matrix must have the same shape")
        if real_matrix.ndim != 2:
            raise ValueError("Inputs must be 2D matrices")

        self.real: np.ndarray = np.asarray(real_matrix, dtype=float)
        self.synthetic: np.ndarray = np.asarray(synthetic_matrix, dtype=float)
        self.n_rows, self.n_cols = self.real.shape
        self.max_rank_allowed = min(self.n_rows, self.n_cols)

        self.real_col_means = np.nanmean(self.real, axis=0)
        self.real_col_stds = np.nanstd(self.real, axis=0)
        self.real_col_stds_safe = np.copy(self.real_col_stds)
        self.real_col_stds_safe = np.where(self.real_col_stds <= 1, 1, self.real_col_stds)
        self.synthetic_col_means = np.nanmean(self.synthetic, axis=0)
        self.synthetic_col_stds = np.nanstd(self.synthetic, axis=0)
        self.synthetic_col_stds_safe = np.copy(self.synthetic_col_stds)
        self.synthetic_col_stds_safe = np.where(self.synthetic_col_stds <= 1, 1, self.synthetic_col_stds)
        self.real_normalized = (self.real - self.real_col_means) / self.real_col_stds_safe
        self.synthetic_normalized = (self.synthetic - self.synthetic_col_means) / self.synthetic_col_stds_safe

        self.real_diagnostics = None
        self.synthetic_diagnostics = None
        self.stacked_diagnostics = None

        self.dataset_name = dataset_name
        self.nanmask = np.isnan(self.real)

        self.results_dir = os.path.join(PROCESS_AND_DIAGNOSTICS_DIR, dataset_name)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.results_figures_dir = os.path.join(self.results_dir, 'figures')
        if not os.path.exists(self.results_figures_dir):
            os.makedirs(self.results_figures_dir)

    def _save_figure_pdf(self, fig: plt.Figure, filename: str, show: bool = True) -> str:
        """Save figure as PDF in output directory and optionally display it."""
        if not filename.endswith(".pdf"):
            filename = f"{filename}.pdf"
        save_path = os.path.join(self.results_figures_dir, filename)
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        print(f"Saved figure to {save_path}")
        return save_path

    @staticmethod
    def hard_impute_svd(
        X: np.ndarray,
        rank: int = 5,
        max_iter: int = 1000,
        tol: float = 1e-4,
        verbose: bool = False
    ) -> np.ndarray:
        """Perform hard imputation using SVD with rank constraint.

        Implements the hard imputation algorithm: iteratively performs low-rank SVD
        approximation and replaces missing values with the approximation, while keeping
        observed values fixed. The algorithm converges when the change between iterations
        falls below the tolerance threshold.

        Args:
            X: Input matrix with missing values (NaNs), shape (n_rows, n_cols).
            rank: Target rank for matrix completion. Must be positive and <= min(n_rows, n_cols).
                 Default: 5.
            max_iter: Maximum number of iterations. Default: 1000.
            tol: Convergence tolerance. Algorithm stops when relative Frobenius norm change
                between iterations is below this value. Default: 1e-4.
            verbose: If True, print convergence information every 10 iterations. Default: False.

        Returns:
            Completed matrix with missing values filled, same shape as X.

        Note:
            Missing values are initially filled with column means. Columns with all NaNs
            are filled with zeros.
        """
        X_filled = X.copy()
        col_means = np.nanmean(X, axis=0)

        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X_filled[mask, j] = col_means[j]

        # Handle columns with all NaNs
        X_filled = np.nan_to_num(X_filled, nan=0)

        prev_X = X_filled.copy()

        for iteration in range(max_iter):
            X_filled = SVD(X_filled, min(min(X_filled.shape), rank))

            mask_observed = ~np.isnan(X)
            X_filled[mask_observed] = X[mask_observed]

            diff = np.linalg.norm(X_filled - prev_X, 'fro') / np.linalg.norm(prev_X, 'fro')
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration + 1}, Diff: {diff}")
            if diff < tol:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
                
            prev_X = X_filled.copy()
        
        return X_filled

    @staticmethod
    def compute_svd_diagnostics(X: np.ndarray, max_rank: Optional[int] = None) -> Dict[str, Any]:
        """Compute comprehensive SVD diagnostics including singular values and variance explained.
        
        Performs full SVD decomposition and computes various diagnostic metrics including
        singular values, cumulative variance explained, and the ranks needed to explain
        common variance thresholds (90%, 95%, 99%). Missing values are filled with column
        means before SVD computation.

        Args:
            X: Input matrix, shape (n_rows, n_cols). NaN values will be filled with column means.
            max_rank: Maximum rank to analyze. If None, uses min(n_rows, n_cols).
                     If provided, truncates results to this rank. Default: None.
            
        Returns:
            Dictionary containing:
                - 'singular_values': Array of singular values up to max_rank (np.ndarray)
                - 'variance_explained': Cumulative variance explained by each rank (np.ndarray)
                - 'variance_explained_ratio': Proportion of variance at each rank (np.ndarray)
                - 'rank_90': Rank needed to explain 90% variance (int)
                - 'rank_95': Rank needed to explain 95% variance (int)
                - 'rank_99': Rank needed to explain 99% variance (int)
                - 'U': Left singular vectors, shape (n_rows, max_rank) (np.ndarray)
                - 'Vt': Right singular vectors (transposed), shape (max_rank, n_cols) (np.ndarray)
                - 'total_variance': Total variance of the matrix (float)
        """
        # Fill NaN values if present
        if np.any(np.isnan(X)):
            X_filled = X.copy()
            col_means = np.nanmean(X, axis=0)
            for j in range(X.shape[1]):
                mask = np.isnan(X[:, j])
                X_filled[mask, j] = col_means[j]
            X_filled = np.nan_to_num(X_filled, nan=0)
        else:
            X_filled = X
        
        # Perform full SVD
        U, s, Vt = np.linalg.svd(X_filled, full_matrices=False)
        
        if max_rank is None:
            max_rank = len(s)
        else:
            max_rank = min(max_rank, len(s))

        singular_values = s[:max_rank]

        # Compute variance explained
        total_variance = np.sum(s**2)
        if total_variance == 0:
            variance_explained = np.zeros_like(s[:max_rank])
            individual_variance = np.zeros_like(s[:max_rank])
        else:
            cumulative_variance = np.cumsum(s[:max_rank]**2)
            variance_explained = cumulative_variance / total_variance
            individual_variance = s[:max_rank]**2 / total_variance

        rank_90 = np.argmax(variance_explained >= 0.90) + 1 if np.any(variance_explained >= 0.90) else max_rank
        rank_95 = np.argmax(variance_explained >= 0.95) + 1 if np.any(variance_explained >= 0.95) else max_rank
        rank_99 = np.argmax(variance_explained >= 0.99) + 1 if np.any(variance_explained >= 0.99) else max_rank
        
        return {
            'singular_values': singular_values,
            'variance_explained': variance_explained,
            'variance_explained_ratio': individual_variance,
            'rank_90': rank_90,
            'rank_95': rank_95,
            'rank_99': rank_99,
            'U': U[:, :max_rank],
            'Vt': Vt[:max_rank, :],
            'total_variance': total_variance
        }

    @staticmethod
    def select_optimal_rank_cv(
        X: np.ndarray,
        max_rank: Optional[int] = None,
        rank_step: int = 3,
        holdout_fraction: float = 0.2,
        verbose: bool = False
    ) -> Tuple[int, float]:
        """Select optimal rank using cross-validation with holdout validation.

        Uses a holdout validation approach to select the optimal rank for matrix completion.
        The method tests a range of ranks and selects the one with the lowest validation error.

        Args:
            X: Matrix for completion with missing values (NaNs), shape (n_rows, n_cols).
            max_rank: Maximum rank to test. If None, defaults to min(n, m) // 5.
                     If outside [1, min(n, m)], will be clamped to valid range. Default: None.
            rank_step: Step size for rank range (e.g., 3 means testing ranks 1, 4, 7, ...).
                      Default: 3.
            holdout_fraction: Fraction of observed values to hold out for validation.
                            Must be in (0, 1). Default: 0.2.
            verbose: If True, print progress for each rank tested. Default: False.

        Returns:
            Tuple of (optimal_rank, best_error):
                - optimal_rank: The rank with the lowest validation error (int)
                - best_error: The validation error for the optimal rank (float)

        Note:
            Uses random holdout, so results may vary between runs. For reproducibility,
            set numpy random seed before calling.
        """
        n, m = X.shape
        if max_rank is None:
            max_rank = min(n, m) // 5
        elif max_rank > min(n, m) or max_rank < 1:
            print(f"max_rank is outside of allowed range [1, min(n, m)], setting max_rank to min(n, m)")
            max_rank = min(n, m)

        rank_range = list(range(1, max_rank + 1, rank_step))

        observed_mask = ~np.isnan(X)
        holdout_mask = np.random.rand(n, m) < holdout_fraction
        validation_mask = observed_mask & holdout_mask

        X_observed = X.copy()
        X_observed[validation_mask] = np.nan

        best_error = np.inf
        best_rank = None

        errors = []
        for rank in rank_range:
            Y_completed = ProcessAndDiagnostics.hard_impute_svd(X_observed, rank=rank, verbose=False)

            denom = np.linalg.norm(X[validation_mask])
            if denom == 0:
                # If all validation values are zero, fallback to absolute error
                error = np.linalg.norm(Y_completed[validation_mask] - X[validation_mask])
            else:
                error = np.linalg.norm(Y_completed[validation_mask] - X[validation_mask]) / denom
            errors.append(error)

            if error < best_error:
                best_error = error
                best_rank = rank

            if verbose:
                print(f"Rank: {rank}, Error: {error:.6f}")
        
        if verbose:
            print(f"\nResults:")
            print(f"  Optimal rank: {best_rank}")
            print(f"  Best error: {best_error:.6f}")
            print(f"  Error reduction: {(max(errors) - best_error)/max(errors)*100:.1f}%")
        
        return best_rank, best_error
    
    @staticmethod
    def principal_angles_row_spaces(VTA, VTB, rA, rB, plot=False):
        """Compute principal angles between row spaces of two matrices.

        Args:
            VTA (np.ndarray): Right singular vectors of matrix A (transposed).
            VTB (np.ndarray): Right singular vectors of matrix B (transposed).
            rA (int): Rank to use for matrix A.
            rB (int): Rank to use for matrix B.
            plot (bool, optional): Whether to plot the results. Default: False.

        Returns:
            np.ndarray: Singular values representing cosines of principal angles.
        """
        VTA = VTA[:rA, :] # rA x m
        VTB = VTB[:rB, :] # rB x m
        sv = np.linalg.svd(VTA @ VTB.T, compute_uv=False)
        if plot:
            plt.plot(sv, marker="o")
            plt.title("Cosine of principal angles (close to 1 is good)")
            plt.xlabel("Rank")
            plt.ylabel("Cosine of principal angle")
            plt.show()
        return sv

    @staticmethod
    def principal_angles_column_spaces(U_real, U_syn, r_real, r_syn, plot=False):
        """Compute principal angles between column spaces of two matrices.

        Args:
            U_real (np.ndarray): Left singular vectors of real matrix.
            U_syn (np.ndarray): Left singular vectors of synthetic matrix.
            r_real (int): Rank to use for real matrix.
            r_syn (int): Rank to use for synthetic matrix.
            plot (bool, optional): Whether to plot the results. Default: False.

        Returns:
            np.ndarray: Singular values representing cosines of principal angles.
        """
        UA = U_real[:, :r_real] # n x r_real
        UB = U_syn[:, :r_syn] # n x r_syn
        sv = np.linalg.svd(UA.T @ UB, compute_uv=False)
        if plot:
            plt.plot(sv, marker="o")
            plt.title("Cosine of principal angles (close to 1 is good)")
            plt.xlabel("Rank")
            plt.ylabel("Cosine of principal angle")
            plt.show()
        return sv

    @staticmethod
    def projected_frobenius_norm_row_spaces(VTA, VTB, rA, rB):
        """Compute projected Frobenius norm between row spaces of two matrices.

        Args:
            VTA (np.ndarray): Right singular vectors of matrix A (transposed).
            VTB (np.ndarray): Right singular vectors of matrix B (transposed).
            rA (int): Rank to use for matrix A.
            rB (int): Rank to use for matrix B.

        Returns:
            float: Frobenius norm of the difference between projection matrices.
        """
        VTA = VTA[:rA, :] # rA x m
        VTB = VTB[:rB, :] # rB x m
        PA = VTA.T @ VTA
        PB = VTB.T @ VTB
        return np.linalg.norm(PA - PB, "fro")

    @staticmethod
    def projected_frobenius_norm_column_spaces(U_real, U_syn, r_real, r_syn):
        """Compute projected Frobenius norm between column spaces of two matrices.

        Args:
            U_real (np.ndarray): Left singular vectors of real matrix.
            U_syn (np.ndarray): Left singular vectors of synthetic matrix.
            r_real (int): Rank to use for real matrix.
            r_syn (int): Rank to use for synthetic matrix.

        Returns:
            float: Frobenius norm of the difference between projection matrices.
        """
        UA = U_real[:, :r_real] # n x r_real
        UB = U_syn[:, :r_syn] # n x r_syn
        PA = UA @ UA.T
        PB = UB @ UB.T
        return np.linalg.norm(PA - PB, "fro")

    def impute_both_matrices_cv(
        self,
        max_rank: Optional[int] = None,
        rank_step: int = 3,
        holdout_fraction: float = 0.2,
        check_stacked: bool = False,
        verbose: bool = False
    ) -> None:
        """Impute both real and synthetic matrices using cross-validation to select optimal ranks.

        Uses cross-validation to find optimal ranks for imputing both the real and synthetic
        matrices. Optionally also computes optimal rank for the stacked matrix (real and
        synthetic concatenated vertically). The completed matrices are stored as instance
        attributes in both normalized and original scales.

        Args:
            max_rank: Maximum rank to test. If None, uses default from select_optimal_rank_cv.
                     Default: None.
            rank_step: Step size for rank range in cross-validation. Default: 3.
            holdout_fraction: Fraction of observed values to hold out for validation.
                            Must be in (0, 1). Default: 0.2.
            check_stacked: If True, also compute optimal rank for stacked matrix
                          (real and synthetic concatenated). Default: False.
            verbose: If True, print progress for each rank tested. Default: False.

        Returns:
            None. Results are stored as instance attributes:
                - best_rank_real: Optimal rank for real matrix (int)
                - best_rank_synthetic: Optimal rank for synthetic matrix (int)
                - best_rank_stacked: Optimal rank for stacked matrix (int, if check_stacked=True)
                - real_completed_normalized: Completed real matrix in normalized space (np.ndarray)
                - synthetic_completed_normalized: Completed synthetic matrix in normalized space (np.ndarray)
                - real_completed: Completed real matrix in original scale (np.ndarray)
                - synthetic_completed: Completed synthetic matrix in original scale (np.ndarray)
        """
        print(f"Calculating best rank for real matrix...")
        best_rank_real, _ = self.select_optimal_rank_cv(self.real_normalized, max_rank=max_rank, rank_step=rank_step, holdout_fraction=holdout_fraction, verbose=verbose)
        print(f"\nCalculating best rank for synthetic matrix...")
        best_rank_synthetic, _ = self.select_optimal_rank_cv(self.synthetic_normalized, max_rank=max_rank, rank_step=rank_step, holdout_fraction=holdout_fraction, verbose=verbose)
        if check_stacked:
            print(f"\nCalculating best rank for stacked matrix...")
            best_rank_stacked, _ = self.select_optimal_rank_cv(np.vstack([self.real_normalized, self.synthetic_normalized]), max_rank=max_rank, rank_step=rank_step, holdout_fraction=holdout_fraction, verbose=verbose)
        if verbose:
            print(f"\nBest rank for real: {best_rank_real}")
            print(f"Best rank for synthetic: {best_rank_synthetic}")
            if check_stacked:
                print(f"Best rank for stacked: {best_rank_stacked}")
        self.best_rank_real = best_rank_real
        self.best_rank_synthetic = best_rank_synthetic
        if check_stacked:
            self.best_rank_stacked = best_rank_stacked
        self.real_completed_normalized = self.hard_impute_svd(self.real_normalized, rank=best_rank_real)
        self.synthetic_completed_normalized = self.hard_impute_svd(self.synthetic_normalized, rank=best_rank_synthetic)
        self.real_completed = self.real_completed_normalized * self.real_col_stds_safe + self.real_col_means
        self.synthetic_completed = self.synthetic_completed_normalized * self.synthetic_col_stds_safe + self.synthetic_col_means
    
    def svd_diagnostics_both_matrices(self, max_rank: Optional[int] = None, fig: bool = True) -> None:
        """Compute and visualize SVD diagnostics for real, synthetic, and stacked matrices.

        Computes SVD diagnostics for the completed real, synthetic, and stacked matrices,
        then visualizes singular values and cumulative variance explained. The stacked
        matrix is formed by concatenating real and synthetic matrices vertically.

        Args:
            max_rank: Maximum rank to analyze. If None, uses full rank. Default: None.

        Returns:
            None. Results are stored as instance attributes:
                - real_diagnostics: Dictionary of SVD diagnostics for real matrix
                - synthetic_diagnostics: Dictionary of SVD diagnostics for synthetic matrix
                - stacked_diagnostics: Dictionary of SVD diagnostics for stacked matrix

        Raises:
            ValueError: If impute_both_matrices_cv() has not been called first.

        Note:
            Displays two plots:
            1. Singular values (scree plot) on log scale
            2. Cumulative variance explained with threshold lines at 90%, 95%, 99%
            Plots are saved to self.results_figures_dir
        """
        # Ensure imputation has been run
        if not hasattr(self, 'real_completed_normalized') or self.real_completed_normalized is None:
            raise ValueError("Call impute_both_matrices_cv() before svd_diagnostics_both_matrices().")
        if not hasattr(self, 'synthetic_completed_normalized') or self.synthetic_completed_normalized is None:
            raise ValueError("Call impute_both_matrices_cv() before svd_diagnostics_both_matrices().")

        self.real_diagnostics = self.compute_svd_diagnostics(self.real_completed_normalized, max_rank=max_rank)
        self.synthetic_diagnostics = self.compute_svd_diagnostics(self.synthetic_completed_normalized, max_rank=max_rank)
        self.stacked_diagnostics = self.compute_svd_diagnostics(np.vstack([self.real_completed_normalized, self.synthetic_completed_normalized]), max_rank=max_rank)
        
        singular_values_real = self.real_diagnostics['singular_values']
        singular_values_synthetic = self.synthetic_diagnostics['singular_values']
        singular_values_stacked = self.stacked_diagnostics['singular_values']
        variance_explained_real = self.real_diagnostics['variance_explained']
        variance_explained_synthetic = self.synthetic_diagnostics['variance_explained']
        variance_explained_stacked = self.stacked_diagnostics['variance_explained']
        
        # Plot 1: Singular values (scree plot)
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(singular_values_real, 'o-', linewidth=1.5, markersize=3, label='Real')
        ax1.plot(singular_values_synthetic, 'o-', linewidth=1.5, markersize=3, label='Synthetic')
        ax1.plot(singular_values_stacked, 'o-', linewidth=1.5, markersize=3, label='Stacked')
        ax1.set_xlabel('Rank', fontsize=18)
        ax1.set_ylabel('Singular Value (log scale)', fontsize=18)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.legend(fontsize=16)
        ax1.tick_params(labelsize=14)
        plt.tight_layout()
        self._save_figure_pdf(fig1, 'svd_diagnostics_singular_values.pdf', show=fig)

        # Plot 2: Cumulative variance explained
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(variance_explained_real * 100, 'o-', linewidth=1.5, markersize=3, label='Real')
        ax2.plot(variance_explained_synthetic * 100, 'o-', linewidth=1.5, markersize=3, label='Synthetic')
        ax2.plot(variance_explained_stacked * 100, 'o-', linewidth=1.5, markersize=3, label='Stacked')
        ax2.set_xlabel('Rank', fontsize=20)
        ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=20)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=16)
        ax2.set_ylim([0, 105])
        ax2.tick_params(labelsize=14)
        plt.tight_layout()
        self._save_figure_pdf(fig2, 'svd_diagnostics_variance_explained.pdf', show=fig)
    
    def _random_matrices_for_row_and_column_space_diagnostics(self):
        """Generate random matrices for row and column space diagnostics.
        """
        rng = np.random.default_rng(seed=42)
        random_matrix = rng.multivariate_normal(mean=np.zeros(self.n_rows), cov=np.eye(self.n_rows), size=self.n_cols).T

        random_matrix_shuffled = np.copy(self.real_completed_normalized)
        for i in range(self.n_cols):
            random_matrix_shuffled[:, i] = np.random.permutation(random_matrix_shuffled[:, i])
        return random_matrix, random_matrix_shuffled

    def _prepare_two_matrix_row_space_diagnostics(self):
        """Prepare matrices for row space diagnostics.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing:
                - 'real_Vt': Right singular vectors of real matrix.
                - 'synthetic_Vt': Right singular vectors of synthetic matrix.
                - 'Vt_random': Right singular vectors of random matrix.
                - 'Vt_random_shuffled': Right singular vectors of random matrix shuffled.
        """
        if self.real_diagnostics is None or self.synthetic_diagnostics is None:
            raise ValueError("Real and synthetic diagnostics must be computed first")

        real_Vt = self.real_diagnostics['Vt'] # shape min(n_rows, n_cols) x n_cols
        synthetic_Vt = self.synthetic_diagnostics['Vt']
        random_matrix, random_matrix_shuffled = self._random_matrices_for_row_and_column_space_diagnostics()
        _, _, Vt_random = np.linalg.svd(random_matrix, full_matrices=False)
        _, _, Vt_random_shuffled = np.linalg.svd(random_matrix_shuffled, full_matrices=False)

        return {
            'real_Vt': real_Vt,
            'synthetic_Vt': synthetic_Vt,
            'Vt_random': Vt_random,
            'Vt_random_shuffled': Vt_random_shuffled
        }
    
    def two_matrix_row_spaces_diagnostics_p_angles(self, max_rank=None, fig: bool = True):
        """Compare row spaces between real, synthetic, and random baselines using principal angles.

        Args:
            max_rank (int, optional): Maximum rank to use for comparison. Default: None.
            fig (bool, optional): Whether to plot the results. Default: True.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing:
                - 'p_angle_syn_real': Principal angles between synthetic and real row spaces.
                - 'p_angle_syn_random': Principal angles between random and real row spaces.
                - 'p_angle_syn_random_shuffled': Principal angles between random shuffled and real row spaces.
        """
        diag_dict = self._prepare_two_matrix_row_space_diagnostics()
        real_Vt = diag_dict['real_Vt']
        synthetic_Vt = diag_dict['synthetic_Vt']
        Vt_random = diag_dict['Vt_random']
        Vt_random_shuffled = diag_dict['Vt_random_shuffled']
        max_rank = min(max_rank, self.max_rank_allowed)

        p_angle_syn_real = self.principal_angles_row_spaces(real_Vt, synthetic_Vt, max_rank, max_rank, plot=False)
        p_angle_syn_random = self.principal_angles_row_spaces(real_Vt, Vt_random, max_rank, max_rank, plot=False)
        p_angle_syn_random_shuffled = self.principal_angles_row_spaces(real_Vt, Vt_random_shuffled, max_rank, max_rank, plot=False)

        p_angle_syn_real_mean = np.mean(p_angle_syn_real)
        p_angle_syn_random_mean = np.mean(p_angle_syn_random)
        p_angle_syn_random_shuffled_mean = np.mean(p_angle_syn_random_shuffled)

        print("----------Mean cosine of principal angles----------")
        print(f"Synthetic vs Real: {p_angle_syn_real_mean:.4f}")
        print(f"Random vs Real: {p_angle_syn_random_mean:.4f}")
        print(f"Random Shuffled vs Real: {p_angle_syn_random_shuffled_mean:.4f}")
        fig_obj, ax = plt.subplots(figsize=(8, 6))
        ax.plot(p_angle_syn_real, marker="o", label="Synthetic vs Real")
        ax.plot(p_angle_syn_random, marker="o", label="Random vs Real")
        ax.plot(p_angle_syn_random_shuffled, marker="o", label="Random Shuffled vs Real")
        ax.set_xlabel("Rank", fontsize=20)
        ax.set_ylabel("Cosine of principal angle", fontsize=20)
        ax.legend(fontsize=16)
        ax.tick_params(labelsize=14)
        plt.tight_layout()
        self._save_figure_pdf(fig_obj, 'row_spaces_p_angles.pdf', show=fig)
        return {
            'p_angle_syn_real': p_angle_syn_real,
            'p_angle_syn_random': p_angle_syn_random,
            'p_angle_syn_random_shuffled': p_angle_syn_random_shuffled
        }
    
    def two_matrix_row_spaces_diagnostics_p_frob_norm(self, max_rank=None, normalize_by_rank=True, fig: bool = True):
        """Compare row spaces between real, synthetic, and random baselines using projected Frobenius norm.

        Args:
            max_rank (int, optional): Maximum rank to use for comparison. Default: None.
            normalize_by_rank (bool, optional): Whether to normalize the projected Frobenius norm by square root of rank. Default: True.
            fig (bool, optional): Whether to plot the results. Default: True.

        Returns:
            Dict[str, list]: Dictionary containing:
                - 'ranks': List of ranks.
                - 'p_frob_norm_syn_real': List of projected Frobenius norms for synthetic vs real.
                - 'p_frob_norm_random': List of projected Frobenius norms for random vs real.
                - 'p_frob_norm_random_shuffled': List of projected Frobenius norms for random shuffled vs real.
        """
        diag_dict = self._prepare_two_matrix_row_space_diagnostics()
        real_Vt = diag_dict['real_Vt']
        synthetic_Vt = diag_dict['synthetic_Vt']
        Vt_random = diag_dict['Vt_random']
        Vt_random_shuffled = diag_dict['Vt_random_shuffled']
        max_rank = min(max_rank, self.max_rank_allowed)

        ranks = list(range(1, max_rank + 1))
        p_frob_norm_syn_real = []
        p_frob_norm_random = []
        p_frob_norm_random_shuffled = []
        norm_factors = {rank: np.sqrt(rank) if normalize_by_rank else 1 for rank in ranks}
        for rank in ranks:
            p_frob_norm_syn_real.append(self.projected_frobenius_norm_row_spaces(real_Vt, synthetic_Vt, rank, rank) / norm_factors[rank])
            p_frob_norm_random.append(self.projected_frobenius_norm_row_spaces(real_Vt, Vt_random, rank, rank) / norm_factors[rank])
            p_frob_norm_random_shuffled.append(self.projected_frobenius_norm_row_spaces(real_Vt, Vt_random_shuffled, rank, rank) / norm_factors[rank])
        fig_obj, ax = plt.subplots(figsize=(8, 6))
        ax.plot(ranks, p_frob_norm_syn_real, marker="o", label="Synthetic vs Real")
        ax.plot(ranks, p_frob_norm_random, marker="o", label="Random vs Real")
        ax.plot(ranks, p_frob_norm_random_shuffled, marker="o", label="Random Shuffled vs Real")
        ax.set_xlabel("Rank", fontsize=20)
        ax.set_ylabel("Projected Frobenius norm\n(close to 0 is good, normalized by rank)" if normalize_by_rank else "Projected Frobenius norm\n(close to 0 is good)", fontsize=20)
        ax.legend(fontsize=16)
        ax.tick_params(labelsize=14)
        plt.tight_layout()
        self._save_figure_pdf(fig_obj, 'row_spaces_p_frob_norm.pdf', show=fig)
        return {
            'ranks': ranks,
            'p_frob_norm_syn_real': p_frob_norm_syn_real,
            'p_frob_norm_random': p_frob_norm_random,
            'p_frob_norm_random_shuffled': p_frob_norm_random_shuffled
        }

    def _prepare_two_matrix_column_space_diagnostics(self):
        """Prepare matrices for column space diagnostics.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing:
                - 'real_U': Left singular vectors of real matrix.
                - 'synthetic_U': Left singular vectors of synthetic matrix.
                - 'U_random': Left singular vectors of random matrix.
                - 'U_random_shuffled': Left singular vectors of random matrix shuffled.
        """
        if self.real_diagnostics is None or self.synthetic_diagnostics is None:
            raise ValueError("Real and synthetic diagnostics must be computed first")
        
        real_U = self.real_diagnostics['U'] # shape n_rows x min(n_rows, n_cols)
        synthetic_U = self.synthetic_diagnostics['U']
        random_matrix, random_matrix_shuffled = self._random_matrices_for_row_and_column_space_diagnostics()
        U_random, _, _ = np.linalg.svd(random_matrix, full_matrices=False)
        U_random_shuffled, _, _ = np.linalg.svd(random_matrix_shuffled, full_matrices=False)

        return {
            'real_U': real_U,
            'synthetic_U': synthetic_U,
            'U_random': U_random,
            'U_random_shuffled': U_random_shuffled
        }
    
    def two_matrix_column_spaces_diagnostics_p_angles(self, max_rank=None, fig: bool = True):
        """Compare column spaces between real, synthetic, and random baselines using principal angles.

        Args:
            max_rank (int, optional): Maximum rank to use for comparison. Default: None.
            fig (bool, optional): Whether to plot the results. Default: True.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing:
                - 'p_angle_syn_real': Principal angles between synthetic and real column spaces.
                - 'p_angle_syn_random': Principal angles between random and real column spaces.
                - 'p_angle_syn_random_shuffled': Principal angles between random shuffled and real column spaces.
        """

        diag_dict = self._prepare_two_matrix_column_space_diagnostics()
        real_U = diag_dict['real_U']
        synthetic_U = diag_dict['synthetic_U']
        U_random = diag_dict['U_random']
        U_random_shuffled = diag_dict['U_random_shuffled']
        max_rank = min(max_rank, self.max_rank_allowed)

        p_angle_syn_real = self.principal_angles_column_spaces(real_U, synthetic_U, max_rank, max_rank, plot=False)
        p_angle_syn_random = self.principal_angles_column_spaces(real_U, U_random, max_rank, max_rank, plot=False)
        p_angle_syn_random_shuffled = self.principal_angles_column_spaces(real_U, U_random_shuffled, max_rank, max_rank, plot=False)

        p_angle_syn_real_mean = np.mean(p_angle_syn_real)
        p_angle_syn_random_mean = np.mean(p_angle_syn_random)
        p_angle_syn_random_shuffled_mean = np.mean(p_angle_syn_random_shuffled)

        print("----------Mean cosine of principal angles----------")
        print(f"Synthetic vs Real: {p_angle_syn_real_mean:.4f}")
        print(f"Random vs Real: {p_angle_syn_random_mean:.4f}")
        print(f"Random Shuffled vs Real: {p_angle_syn_random_shuffled_mean:.4f}")
        fig_obj, ax = plt.subplots(figsize=(8, 6))
        ax.plot(p_angle_syn_real, marker="o", label="Synthetic vs Real")
        ax.plot(p_angle_syn_random, marker="o", label="Random vs Real")
        ax.plot(p_angle_syn_random_shuffled, marker="o", label="Random Shuffled vs Real")
        ax.set_xlabel("Rank", fontsize=20)
        ax.set_ylabel("Cosine of principal angle", fontsize=20)
        ax.legend(fontsize=16)
        ax.tick_params(labelsize=14)
        plt.tight_layout()
        self._save_figure_pdf(fig_obj, 'column_spaces_p_angles.pdf', show=fig)
        return {
            'p_angle_syn_real': p_angle_syn_real,
            'p_angle_syn_random': p_angle_syn_random,
            'p_angle_syn_random_shuffled': p_angle_syn_random_shuffled
        }
        
    def two_matrix_column_spaces_diagnostics_p_frob_norm(self, max_rank=None, normalize_by_rank=True, fig: bool = True):
        """Compare column spaces between real, synthetic, and random baselines using projected Frobenius norm.

        Args:
            max_rank (int, optional): Maximum rank to use for comparison. Default: None.
            normalize_by_rank (bool, optional): Whether to normalize the projected Frobenius norm by square root of rank. Default: True.
            fig (bool, optional): Whether to plot the results. Default: True.

        Returns:
            Dict[str, list]: Dictionary containing:
                - 'ranks': List of ranks.
                - 'p_frob_norm_syn_real': List of projected Frobenius norms for synthetic vs real.
                - 'p_frob_norm_random': List of projected Frobenius norms for random vs real.
                - 'p_frob_norm_random_shuffled': List of projected Frobenius norms for random shuffled vs real.
        """
        diag_dict = self._prepare_two_matrix_column_space_diagnostics()
        real_U = diag_dict['real_U']
        synthetic_U = diag_dict['synthetic_U']
        U_random = diag_dict['U_random']
        U_random_shuffled = diag_dict['U_random_shuffled']
        max_rank = min(max_rank, self.max_rank_allowed)

        ranks = list(range(1, max_rank + 1))
        p_frob_norm_syn_real = []
        p_frob_norm_random = []
        p_frob_norm_random_shuffled = []
        norm_factors = {rank: np.sqrt(rank) if normalize_by_rank else 1 for rank in ranks}
        for rank in ranks:
            p_frob_norm_syn_real.append(self.projected_frobenius_norm_column_spaces(real_U, synthetic_U, rank, rank) / norm_factors[rank])
            p_frob_norm_random.append(self.projected_frobenius_norm_column_spaces(real_U, U_random, rank, rank) / norm_factors[rank])
            p_frob_norm_random_shuffled.append(self.projected_frobenius_norm_column_spaces(real_U, U_random_shuffled, rank, rank) / norm_factors[rank])
        fig_obj, ax = plt.subplots(figsize=(8, 6))
        ax.plot(ranks, p_frob_norm_syn_real, marker="o", label="Synthetic vs Real")
        ax.plot(ranks, p_frob_norm_random, marker="o", label="Random vs Real")
        ax.plot(ranks, p_frob_norm_random_shuffled, marker="o", label="Random Shuffled vs Real")
        ax.set_xlabel("Rank", fontsize=20)
        ax.set_ylabel("Projected Frobenius norm\n(close to 0 is good, normalized by rank)" if normalize_by_rank else "Projected Frobenius norm\n(close to 0 is good)", fontsize=20)
        ax.legend(fontsize=16)
        ax.tick_params(labelsize=14)
        plt.tight_layout()
        self._save_figure_pdf(fig_obj, 'column_spaces_p_frob_norm.pdf', show=fig)

        return {
            'ranks': ranks,
            'p_frob_norm_syn_real': p_frob_norm_syn_real,
            'p_frob_norm_random': p_frob_norm_random,
            'p_frob_norm_random_shuffled': p_frob_norm_random_shuffled
        }
    
    def process_and_diagnostic(
        self,
        max_rank=None,
        rank_step=3,
        holdout_fraction=0.2,
        check_stacked=True,
        normalize_by_rank=True,
        fig=True,
        verbose=True
    ) -> Dict[str, Any]:
        """Execute the complete diagnostic pipeline: imputation, SVD diagnostics, and space comparisons.
        
        This function runs the full diagnostic workflow:
        1. Imputes both matrices using cross-validation
        2. Computes SVD diagnostics
        3. Compares row spaces using principal angles and Frobenius norm
        4. Compares column spaces using principal angles and Frobenius norm
        
        Args:
            max_rank (int, optional): Maximum rank to test/analyze. Default: None.
            rank_step (int, optional): Step size for rank range in CV. Default: 3.
            holdout_fraction (float, optional): Fraction of observed values to hold out for validation. Default: 0.2.
            check_stacked (bool, optional): Whether to check the effective rank of the stacked matrix. Default: True.
            normalize_by_rank (bool, optional): Whether to normalize the projected Frobenius norm by square root of rank. Default: True.
            fig (bool, optional): Whether to plot the results. Default: True.
            verbose (bool, optional): Whether to print progress. Default: True.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'row_spaces': Dictionary with:
                    - 'p_angles': Results from two_matrix_row_spaces_diagnostics_p_angles
                    - 'p_frob_norm': Results from two_matrix_row_spaces_diagnostics_p_frob_norm
                - 'column_spaces': Dictionary with:
                    - 'p_angles': Results from two_matrix_column_spaces_diagnostics_p_angles
                    - 'p_frob_norm': Results from two_matrix_column_spaces_diagnostics_p_frob_norm
                - 'real_completed': Completed real matrix (after imputation)
                - 'synthetic_completed': Completed synthetic matrix (after imputation)
                - 'nanmask': Boolean mask indicating originally missing values in real matrix
        """
        # Step 1: Impute both matrices using cross-validation
        print("----------Imputing both matrices----------")
        self.impute_both_matrices_cv(
            max_rank=max_rank,
            rank_step=rank_step,
            holdout_fraction=holdout_fraction,
            check_stacked=check_stacked,
            verbose=verbose
        )
        # Use stacked rank + 2 for space diagnostics, but ensure it doesn't exceed max_rank_allowed
        if check_stacked:
            max_rank_for_space_diagnostics = min(self.best_rank_stacked + 2, self.max_rank_allowed)
        else:
            # If stacked rank not computed, use max of real and synthetic ranks
            max_rank_for_space_diagnostics = min(max(self.best_rank_real, self.best_rank_synthetic) + 2, self.max_rank_allowed)
        
        # Step 2: Compute SVD diagnostics
        print("\n----------Computing SVD diagnostics----------")
        self.svd_diagnostics_both_matrices(fig=fig)
        
        # Step 3: Compare row spaces
        print("\n----------Computing two matrix row space diagnostics----------")
        row_p_angles = self.two_matrix_row_spaces_diagnostics_p_angles(max_rank=max_rank_for_space_diagnostics, fig=fig)
        row_p_frob_norm = self.two_matrix_row_spaces_diagnostics_p_frob_norm(
            max_rank=max_rank_for_space_diagnostics,
            normalize_by_rank=normalize_by_rank,
            fig=fig
        )
        
        # Step 4: Compare column spaces
        print("\n----------Computing two matrix column space diagnostics----------")
        col_p_angles = self.two_matrix_column_spaces_diagnostics_p_angles(max_rank=max_rank_for_space_diagnostics, fig=fig)
        col_p_frob_norm = self.two_matrix_column_spaces_diagnostics_p_frob_norm(
            max_rank=max_rank_for_space_diagnostics,
            normalize_by_rank=normalize_by_rank,
            fig=fig
        )
        
        # Return comprehensive results dictionary
        return {
            'row_spaces': {
                'p_angles': row_p_angles,
                'p_frob_norm': row_p_frob_norm
            },
            'column_spaces': {
                'p_angles': col_p_angles,
                'p_frob_norm': col_p_frob_norm
            },
            'real_completed': self.real_completed,
            'synthetic_completed': self.synthetic_completed,
            'nanmask': self.nanmask
        }