import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, Any, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

class IDSExplainer:
    """
    Explainable AI Module for IDS using SHAP.
    Provides global insights and local explanations for specific alerts.
    """
    def __init__(self, model, X_sample: Optional[pd.DataFrame] = None, output_dir="models/explanations"):
        """
        Args:
            model: The trained model (XGBClassifier or Booster)
            X_sample: A sample of training data (e.g. 50-100 rows) for background distribution.
                      Required if TreeExplainer fails and we fallback to Kernel/Permutation.
            output_dir: Directory to save plots/jsons.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.X_sample = X_sample
        
        # Check if model is wrapped or raw
        if hasattr(model, "model"):
             # Our wrapper
             self.model = model.model
        else:
             self.model = model
             
        self.explainer = None
        self._initialize_explainer()

    def _initialize_explainer(self):
        # 1. Try TreeExplainer (Optimized)
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP TreeExplainer initialized successfully.")
            return
        except Exception as e:
            logger.warning(f"TreeExplainer initialization failed: {e}")
            
        # 2. Fallback to Generic Explainer (Permutation/Kernel)
        # Needs a callable (predict_proba) and background data
        if self.X_sample is not None:
            try:
                logger.info("Attempting fallback to Generic SAMPLING Explainer (slower)...")
                # Predict probability of class 1
                # If model has predict_proba
                if hasattr(self.model, "predict_proba"):
                    self.explainer = shap.Explainer(self.model.predict_proba, self.X_sample)
                else:
                    self.explainer = shap.Explainer(self.model.predict, self.X_sample)
                logger.info("SHAP Generic Explainer initialized.")
                return
            except Exception as e2:
                logger.error(f"Fallback Explainer failed: {e2}")
        else:
            logger.error("X_sample not provided. Cannot use fallback Explainer.")
            
        if self.explainer is None:
            raise RuntimeError("Could not initialize any SHAP Explainer.")

    def explain_global(self, X: pd.DataFrame, save_plots=True):
        """
        Generate global explanation plots (Summary & Bar).
        """
        if not self.explainer:
             logger.error("Explainer not initialized.")
             return

        logger.info("Generating Global SHAP Plots...")
        try:
            # Calculate values
            # shap_values from generic explainer might be Explanation object
            shap_obj = self.explainer(X)
            
            # Extract values for plots if needed, or pass object depending on plot function
            # shap.summary_plot accepts Explanation object in newer versions
            
            if save_plots:
                # 1. Summary Plot (Dot plot)
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_obj, X, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "shap_summary.png"))
                plt.close()
                
                # 2. Bar Plot (Importance)
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_obj, X, plot_type="bar", show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "shap_importance_bar.png"))
                plt.close()
                
                logger.info(f"Plots saved to {self.output_dir}")
                
        except Exception as e:
            logger.error(f"Error generating global plots: {e}")

    def explain_local(self, row_df: pd.DataFrame, top_n: int = 5) -> Dict[str, Any]:
        """
        Explain a single prediction (alert).
        Returns JSON-compatible Dict.
        """
        try:
            # shap_values returns Explanation object for Generic Explainer usually
            explanation_obj = self.explainer(row_df)
            
            # If it's a list (multiclass), take index 1? Or if binary, index 0?
            # For predict_proba, it usually returns values for all classes.
            # Explanation object shape: (1, n_features, n_classes) or (1, n_features)
            
            vals = explanation_obj.values
            base_val = explanation_obj.base_values
            
            # Handle dimensions
            if len(vals.shape) == 3: 
                # (rows, features, classes) -> Take Class 1 (Attack)
                vals = vals[0, :, 1]
                base_val = base_val[0, 1]
            elif len(vals.shape) == 2:
                # (rows, features)
                vals = vals[0, :]
                base_val = base_val[0]
            elif len(vals.shape) == 1:
                pass # Already 1D
                
            if isinstance(base_val, np.ndarray):
                base_val = float(base_val)
            
            feature_names = row_df.columns
            feature_values = row_df.iloc[0].values
            
            # Create feature dict list
            features_list = []
            for name, shap_val, feat_val in zip(feature_names, vals, feature_values):
                features_list.append({
                    "feature": name,
                    "value": float(feat_val),
                    "impact_score": float(shap_val),
                    "impact_direction": "Risk Increase" if shap_val > 0 else "Risk Decrease",
                    "abs_impact": abs(shap_val)
                })
            
            # Sort by absolute impact
            features_list.sort(key=lambda x: x['abs_impact'], reverse=True)
            top_features = features_list[:top_n]
            
            explanation = {
                "base_value": float(base_val),
                "top_features": top_features,
                "explanation_text": f"Base risk was {float(base_val):.2f}. " + 
                                  " ".join([f"{f['feature']} ({f['impact_direction']})" for f in top_features])
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining local prediction: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def save_explanation(self, explanation: Dict, filename="alert_explanation.json"):
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            json.dump(explanation, f, indent=4)
        logger.info(f"Explanation saved to {path}")
