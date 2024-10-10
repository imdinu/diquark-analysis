from tqdm import tqdm
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from diquark.config.config_manager import ConfigManager
from diquark.utils.logger import setup_logger
from diquark.utils.results_manager import ResultsManager
from diquark.data.loader import DataLoader
from diquark.features.feature_extractor import FeatureExtractor
from diquark.data.preprocessor import Preprocessor
from diquark.models.neural_network import NeuralNetworkModel
from diquark.models.random_forest import RandomForestModel
from diquark.models.gradient_boosting import GradientBoostingModel
from diquark.evaluation.metrics import calculate_metrics, calculate_signal_background_metrics, calculate_counts_for_score_cuts
from diquark.evaluation.visualizations import plot_results
from diquark.config import constants  

class Analysis:
    def __init__(self, config_path: str):
        self.config = ConfigManager(config_path)
        self.logger = setup_logger('experiment_logger', 
                                   self.config.get('logging.file_path', 'logs/experiment.log'),
                                   level=self.config.get('logging.level', 'INFO'))
        self.results_manager = ResultsManager(self.config.get('results.directory', 'results'))

        
        # Load path_dict and cross_sections from constants
        self.path_dict = getattr(constants, self.config.get('data.path_dict'))  # Adjust as needed
        self.cross_sections = getattr(constants, self.config.get('data.cross_section_dict'))  # Adjust as needed

        self.use_cross_validation = self.config.get('cross_validation.enabled', False)
        self.n_folds = self.config.get('cross_validation.n_folds', 1)
        
        self.data_loader = DataLoader(self.path_dict, 
                                      n_jets=self.config.get('data.n_jets', 6),
                                      index_start=self.config.get('data.index_start', 0),
                                      index_stop=self.config.get('data.index_stop', None))
        
        self.feature_extractor = FeatureExtractor(n_jets=self.config.get('feature_extraction.n_jets', 6))
        self.preprocessor = Preprocessor(self.config.get('preprocessing', {}))
        
        self.models = {
            'neural_network': NeuralNetworkModel(self.config.get('models.neural_network', {})) if self.config.get('models.neural_network', None) else None,
            'random_forest': RandomForestModel(self.config.get('models.random_forest', {})) if self.config.get('models.random_forest', None) else None,
            'gradient_boosting': GradientBoostingModel(self.config.get('models.gradient_boosting', {})) if self.config.get('models.gradient_boosting', None) else None
        }

    # def run(self):
    #     self.logger.info("Starting analysis...")
        
    #     # Load data
    #     data = self.load_data()
        
    #     # Extract features
    #     features = self.extract_features(data)
        
    #     # Preprocess data
    #     X_train, X_test, y_train, y_test, df_train, df_test = self.preprocess_data(features)
        
    #     # Train and evaluate models
    #     results = self.train_and_evaluate_models(X_train, X_test, y_train, y_test, df_test)
        
    #     # Visualize results
    #     self.visualize_results(results, df_test)
        
    #     # Save results
    #     self.save_results(results, df_test)
        
    #     self.logger.info("Analysis completed.")
        
    def run(self):
        self.logger.info("Starting analysis...")
        
        # Load data
        data = self.load_data()
        
        # Extract features
        features = self.extract_features(data)
        
        if self.use_cross_validation:
            self.run_cross_validation(features)
        else:
            self.run_single_fold(features)
        
        self.logger.info("Analysis completed.")

    def run_single_fold(self, features):
        # Preprocess data
        X_train, X_test, y_train, y_test, df_train, df_test = self.preprocess_data(features)
        
        # Train and evaluate models
        results = self.train_and_evaluate_models(X_train, X_test, y_train, y_test, df_test)
        
        # Visualize results
        self.visualize_results(results, df_test)
        
        # Save results
        self.save_results(results, df_test)

    def run_cross_validation(self, features):
        df = self.preprocessor.create_dataframe(features)
        X = df.drop(["target", "Truth", "combined_invariant_mass"], axis=1)
        y = df["target"]
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        all_fold_results = []
        
        for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
            self.logger.info(f"Processing fold {fold}")
            
            fold_dir = self.results_manager.create_subdir(f"fold_{fold}")
            
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            df_train = df.iloc[train_index]
            df_test = df.iloc[test_index]
            
            X_train, X_test, y_train, y_test, df_train, df_test = self.preprocessor.prepare_fold_data(X_train, X_test, y_train, y_test, df_train, df_test)
            
            results = self.train_and_evaluate_models(X_train, X_test, y_train, y_test, df_test)
            
            self.visualize_results(results, df_test, fold_dir)
            self.save_results(results, df_test, fold_dir)
            
            all_fold_results.append(results)
        
        self.summarize_cross_validation_results(all_fold_results)

    def load_data(self):
        self.logger.info("Loading data...")
        return self.data_loader.load_data(self.config.get('data.mass_cut'))

    def extract_features(self, data):
        self.logger.info("Extracting features...")
        return {key: self.feature_extractor.compute_all(arr) for key, arr in tqdm(data.items())}

    def preprocess_data(self, features):
        self.logger.info("Preprocessing data...")
        return self.preprocessor.prepare_data(features)

    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test, df_test):
        results = {}
        for model_name, model in self.models.items():
            if model is None:
                continue
            self.logger.info(f"Training and evaluating {model_name}...")
            model.build(X_train.shape[1])
            model.train(X_train, y_train, X_test, y_test)
            y_pred = model.predict(X_test)

            sample_weights = np.array([self.cross_sections[label] for label in df_test['Truth']])
            
            thresholds = self.config.get('evaluation.thresholds', [0.2, 0.5, 0.8, 0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99])
            use_real_event_percentiles = self.config.get('evaluation.use_real_event_percentiles', False)
            total_luminosity = self.config.get('data.total_luminosity', 3000)

            metrics = calculate_metrics(y_test, y_pred, sample_weights, thresholds, use_real_event_percentiles)
            sig_bkg_metrics = calculate_signal_background_metrics(y_test, y_pred, thresholds, sample_weights, total_luminosity, self.cross_sections, use_real_event_percentiles)
            
            cuts = thresholds
            df_counts = calculate_counts_for_score_cuts(y_test, y_pred, df_test['combined_invariant_mass'], df_test['Truth'], self.cross_sections, total_luminosity, cuts, use_real_event_percentiles)

            results[model_name] = {
                'metrics': metrics,
                'sig_bkg_metrics': sig_bkg_metrics,
                'predictions': y_pred,
                'counts': df_counts
            }
            if hasattr(model, 'feature_importances'):
                results[model_name]['feature_importances'] = model.feature_importances()
        return results

    def visualize_results(self, results, df_test, custom_dir=None):
        self.logger.info("Visualizing results...")
        plot_types = self.config.get('visualization.plots', [])
        results_dir = custom_dir if custom_dir else self.config.get('results.directory', 'results')
        for model_name, model_results in results.items():
            plots = plot_results(model_results, df_test, plot_types, self.config.get('evaluation.use_real_event_percentiles', True))
            for plot_name, fig in plots.items():
                fig.write_image(f"{results_dir}/{model_name}_{plot_name}.pdf")

    def save_results(self, results, df_test, custom_dir=None):
        results_dir = custom_dir if custom_dir else self.results_manager.results_dir
        self.logger.info(f"Saving results to {results_dir}...")
        for model_name, model_results in results.items():
            self.results_manager.save_json(model_results['metrics'], f"{model_name}_metrics.json", custom_dir=results_dir)
            self.results_manager.save_json(model_results['sig_bkg_metrics'], f"{model_name}_sig_bkg_metrics.json", custom_dir=results_dir)
            self.results_manager.save_dataframe(model_results['counts'], f"{model_name}_counts.csv", custom_dir=results_dir)
            
            if self.config.get('results.save_predictions', False):
                self.results_manager.save_numpy(model_results['predictions'], f"{model_name}_predictions.npy", custom_dir=results_dir)
            
            if self.config.get('results.save_feature_importances', False) and 'feature_importances' in model_results:
                feature_importances = model_results['feature_importances']
                if len(feature_importances) == len(self.feature_extractor.feature_names):
                    feature_importance_dict = dict(zip(self.feature_extractor.feature_names, feature_importances.tolist()))
                else:
                    self.logger.warning(f"Feature importances length ({len(feature_importances)}) does not match feature names length ({len(self.feature_extractor.feature_names)}). Saving without feature names.")
                    feature_importance_dict = dict(enumerate(feature_importances.tolist()))
                
                self.results_manager.save_json(feature_importance_dict, f"{model_name}_feature_importances.json", custom_dir=results_dir)
        
        # Save test set
        self.results_manager.save_dataframe(df_test, "test_set.parquet", format="parquet", custom_dir=results_dir)

    def summarize_cross_validation_results(self, all_fold_results):
        for model_name in self.models.keys():
            if self.models[model_name] is None:
                continue
            
            dfs = [fold_result[model_name]['counts'] for fold_result in all_fold_results]
            combined_df = pd.concat(dfs, axis=0, keys=[f"Fold {i}" for i in range(1, self.n_folds + 1)])
            
            # Separate numeric and non-numeric columns
            numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
            non_numeric_cols = combined_df.select_dtypes(exclude=[np.number]).columns
            
            # Calculate mean and std for numeric columns
            mean_df = combined_df[numeric_cols].groupby(level=1).mean()
            std_df = combined_df[numeric_cols].groupby(level=1).std()
            
            # For non-numeric columns, use the most common value
            for col in non_numeric_cols:
                mean_df[col] = combined_df[col].groupby(level=1).agg(lambda x: x.value_counts().index[0])
                std_df[col] = ''  # Empty string for std of non-numeric columns
            
            result_df = pd.DataFrame(index=mean_df.index, columns=mean_df.columns)
            
            for col in mean_df.columns:
                for idx in mean_df.index:
                    mean_value = mean_df.loc[idx, col]
                    std_value = std_df.loc[idx, col]
                    if col in numeric_cols:
                        result_df.loc[idx, col] = self.format_value(mean_value, std_value)
                    else:
                        result_df.loc[idx, col] = mean_value
            
            self.results_manager.save_dataframe(result_df, f"{model_name}_counts_summary.csv")

    def format_value(self, mean, std):
        if mean == 0 and std == 0:
            return "0.000e+00 ± 0.000e+00"

        mean_magnitude = int(np.floor(np.log10(abs(mean)))) if mean != 0 else 0
        std_magnitude = int(np.floor(np.log10(abs(std)))) if std != 0 else 0

        precision = max(0, mean_magnitude - std_magnitude + 3)

        mean_formatted = f"{mean:.{precision}e}"
        std_formatted = f"{std:.{precision}e}"

        return f"{mean_formatted} ± {std_formatted}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diquark analysis with optional custom config file.")
    parser.add_argument("-c", "--config", type=str, default='diquark/config/default_settings.yaml',
                        help="Path to the configuration file (default: diquark/config/default_settings.yaml)")
    
    args = parser.parse_args()
    
    analysis = Analysis(args.config)
    analysis.run()