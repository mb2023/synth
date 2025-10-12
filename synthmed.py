import copy
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Dict

# SynthCity
import sys
import synthcity.logger as log
log.add(sink=sys.stderr, level="DEBUG")

from synthcity.plugins.core.dataloader import DataLoader
from synthcity.utils.reproducibility import enable_reproducible_results, clear_cache
from synthcity.metrics import Metrics
from synthcity.plugins import Plugins
from synthcity.utils.serialization import save_to_file, load_from_file


class SynthMed:
    """
    A class to manage synthetic data generation and evaluation using SynthCity models.

    Attributes:
        X (DataLoader): The training data.
        X_test (DataLoader): The test data.
        model (str): The model to use for synthetic data generation.
        epsilon (Optional[str]): The privacy parameter for differential privacy models.
        repeats (int): The number of times to repeat the model training.
        evaluations_folder (Path): The folder to save evaluation results.
        generators_folder (Path): The folder to save generators.
        synthetic_data_folder (Path): The folder to save synthetic data.
        tmp_workspace (Path): The temporary workspace folder.
        path_stub (str): The base path stub for file naming.
        seeds (np.ndarray): The random seeds for reproducibility.
    """

    def __init__(
        self,
        X: DataLoader,
        X_test: DataLoader,
        model: str = "adsgan",
        epsilon: Optional[str] = None,
        repeats: int = 10,
        parent_folder: Optional[Path] = None,
    ):
        """
        Initialize the SynthMed class with data and model parameters.

        Args:
            X (DataLoader): The training data.
            X_test (DataLoader): The test data.
            model (str): The model to use for synthetic data generation. Defaults to ADSGAN.
            epsilon (Optional[str]): The privacy parameter for differential privacy models.
            repeats (int): The number of times to repeat the model training.
            parent_folder (Optional[Path]): The parent folder to save results.
        """
        self.X = X
        self.X_test = X_test
        self.model = model
        self.epsilon = epsilon
        self.repeats = repeats
        self._create_folders(parent_folder)

    def _create_folders(self, parent_folder: Optional[Path]) -> None:
        """
        Create folders to save results, generators, and synthetic data.

        Args:
            parent_folder (Optional[Path]): The parent folder to save results.
        """
        if parent_folder is None:
            parent_folder = Path("../outputs/")

        folders = {
            "evaluations": f"evaluations/{self.model}/",
            "generators": f"generators/{self.model}/",
            "synthetic_data": f"synthetic_data/{self.model}/",
            "tmp_workspace": f"tmp_workspace/{self.model}/",
        }

        for key, folder in folders.items():
            path = parent_folder / folder
            path.mkdir(parents=True, exist_ok=True)
            setattr(self, f"{key}_folder" if key != "tmp_workspace" else key, path)

        self.path_stub = (
            f"{self.model}_eps{self.epsilon}" if self.epsilon else f"{self.model}"
        )

    def run_model(
        self,
        reuse_existing: bool = True,
        kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Run the synthetic data generation model.

        Args:
            reuse_existing (bool): Whether to reuse existing generators.
            kwargs (Optional[Dict]): Additional keyword arguments for the model.
        """
        if kwargs is None:
            kwargs = {}

        if (
            self.model in ["adsgan", "pategan", "dpgan", "privbayes"]
            and not self.epsilon
        ):
            raise ValueError(
                f"Model {self.model.upper()} requires an epsilon (lambda identifiability for ADSGAN) to be specified."
            )

        generators = Plugins()

        if reuse_existing:
            existing_generators = [
                p
                for p in self.generators_folder.iterdir()
                if f"{self.path_stub}_" in str(p)
            ]
            self.repeats -= len(existing_generators)

        rng = np.random.default_rng()
        self.seeds = rng.integers(0, 10000000, self.repeats)

        kws = copy.deepcopy(kwargs)
        kws["workspace"] = self.tmp_workspace

        for rep in range(self.repeats):
            log.info(f"Repeat: {rep} seed: {self.seeds[rep]}")
            enable_reproducible_results(int(self.seeds[rep]))
            clear_cache()

            kws["random_state"] = self.seeds[rep]
            if self.model == "adsgan":
                kws["lambda_identifiability_penalty"] = self.epsilon
            elif self.model in ["pategan", "dpgan", "privbayes"]:
                kws["epsilon"] = self.epsilon

            gen = generators.get(self.model, **kws)
            gen.fit(self.X)
            generator_fname = f"{self.path_stub}_{self.seeds[rep]}_generator.bkp"
            save_to_file(self.generators_folder / generator_fname, gen)

    def deep_generative_ensemble(self) -> None:
        """
        Create a deep generative ensemble from the synthetic data generators.
        """
        generator_paths = [
            p
            for p in self.generators_folder.iterdir()
            if f"{self.path_stub}_" in str(p)
        ]

        seeds = [
            str(p).split("/")[-1].split(f"{self.path_stub}_")[1].split("_")[0]
            for p in generator_paths
        ]

        if len(generator_paths) != len(seeds):
            raise ValueError("Mismatch between number of generators and seeds.")

        generators = {
            seed: load_from_file(p) for seed, p in zip(seeds, generator_paths)
        }

        if len(generators) < self.repeats:
            print(
                "Fewer generators are available than expected. Review existing generators."
            )

        deep_generative_ensemble = [
            generator.generate(count=self.X.data.shape[0]).data
            for generator in generators.values()
        ]

        dge_df_full = pd.concat(deep_generative_ensemble, axis=0)
        dge_df = dge_df_full.sample(
            n=self.X.data.shape[0] + self.X_test.data.shape[0], random_state=42
        )

        dge_df_full.to_csv(
            self.synthetic_data_folder / f"{self.path_stub}_dge_full_synthetic_df.csv"
        )
        dge_df.to_csv(
            self.synthetic_data_folder / f"{self.path_stub}_dge_synthetic_df.csv"
        )

    def evaluate_deep_generative_ensemble(self, eval_metrics: dict = None) -> None:
        """
        Evaluate the deep generative ensemble.
        """
        dge_df = pd.read_csv(
            self.synthetic_data_folder / f"{self.path_stub}_dge_synthetic_df.csv",
            usecols=self.X.data.columns.to_list(),
        )

        if eval_metrics is None:
            eval_metrics = {
                "sanity": [
                    "data_mismatch",
                    "common_rows_proportion",
                    "nearest_syn_neighbor_distance",
                    "close_values_probability",
                    "distant_values_probability",
                ],
                "stats": [
                    "jensenshannon_dist",
                    "chi_squared_test",
                    "inv_kl_divergence",
                    "ks_test",
                    "max_mean_discrepancy",
                    "wasserstein_dist",
                    "prdc",
                    "alpha_precision",
                    "survival_km_distance",
                ],
                "detection": [
                    "detection_xgb",
                    "detection_mlp",
                    "detection_gmm",
                    "detection_linear",
                ],
                "privacy": [
                    "delta-presence",
                    "k-anonymization",
                    "k-map",
                    "distinct l-diversity",
                    "identifiability_score",
                ],
            }
        
        eval_testset = Metrics.evaluate(
            self.X_test,
            dge_df,
            metrics=eval_metrics,
            task_type="survival_analysis",
            random_state=42,
        )

        eval_testset[["mean"]].to_csv(
            self.evaluations_folder / f"{self.path_stub}_evaluation_testset.csv"
        )

        eval_trainset = Metrics.evaluate(
            self.X,
            dge_df,
            metrics=eval_metrics,
            task_type="survival_analysis",
            random_state=42,
        )

        eval_trainset[["mean"]].to_csv(
            self.evaluations_folder / f"{self.path_stub}_evaluation_trainset.csv"
        )
