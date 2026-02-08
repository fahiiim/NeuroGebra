"""
Metrics expression repository.

Pre-built evaluation metric expressions for assessing model performance:
classification metrics, regression metrics, ranking metrics, and
similarity measures. Element-wise or scalar forms suitable for
educational demonstration.
"""

from neurogebra.core.expression import Expression
from typing import Dict


def get_metrics_expressions() -> Dict[str, Expression]:
    """
    Get dictionary of evaluation metric expressions.

    Returns:
        Dictionary mapping names to Expression instances
    """
    met: Dict[str, Expression] = {}

    # ================================================================
    # Classification Metrics (element-wise / scalar forms)
    # ================================================================

    met["accuracy_elem"] = Expression(
        name="accuracy_elem",
        symbolic_expr="Piecewise((1, Eq(y_pred, y_true)), (0, True))",
        metadata={
            "category": "metric",
            "subcategory": "classification",
            "description": "Element-wise accuracy indicator (1 if correct)",
            "usage": "Classification evaluation, per-sample correctness",
            "formula_latex": r"\mathbb{1}[\hat{y}_i = y_i]",
        },
    )

    met["precision_formula"] = Expression(
        name="precision_formula",
        symbolic_expr="tp / (tp + fp + 1e-8)",
        params={"tp": 1, "fp": 0},
        metadata={
            "category": "metric",
            "subcategory": "classification",
            "description": "Precision = TP / (TP + FP)",
            "usage": "When false positives are costly (spam detection)",
            "formula_latex": r"\text{Precision} = \frac{TP}{TP + FP}",
        },
    )

    met["recall_formula"] = Expression(
        name="recall_formula",
        symbolic_expr="tp / (tp + fn + 1e-8)",
        params={"tp": 1, "fn": 0},
        metadata={
            "category": "metric",
            "subcategory": "classification",
            "description": "Recall (Sensitivity) = TP / (TP + FN)",
            "usage": "When false negatives are costly (medical diagnosis)",
            "formula_latex": r"\text{Recall} = \frac{TP}{TP + FN}",
        },
    )

    met["f1_score_formula"] = Expression(
        name="f1_score_formula",
        symbolic_expr="2 * prec * rec / (prec + rec + 1e-8)",
        params={"prec": 1, "rec": 1},
        metadata={
            "category": "metric",
            "subcategory": "classification",
            "description": "F1 score: harmonic mean of precision and recall",
            "usage": "Balanced classification evaluation",
            "formula_latex": r"F_1 = \frac{2 \cdot P \cdot R}{P + R}",
        },
    )

    met["fbeta_score_formula"] = Expression(
        name="fbeta_score_formula",
        symbolic_expr="(1 + beta_f**2) * prec * rec / (beta_f**2 * prec + rec + 1e-8)",
        params={"prec": 1, "rec": 1, "beta_f": 1},
        metadata={
            "category": "metric",
            "subcategory": "classification",
            "description": "Fβ score: weighted harmonic mean (β>1 favors recall)",
            "usage": "Custom precision-recall tradeoff",
            "formula_latex": r"F_\beta = \frac{(1+\beta^2) P R}{\beta^2 P + R}",
        },
    )

    met["specificity_formula"] = Expression(
        name="specificity_formula",
        symbolic_expr="tn / (tn + fp + 1e-8)",
        params={"tn": 1, "fp": 0},
        metadata={
            "category": "metric",
            "subcategory": "classification",
            "description": "Specificity (True Negative Rate) = TN / (TN + FP)",
            "usage": "Complementary to recall for binary classification",
            "formula_latex": r"\text{Specificity} = \frac{TN}{TN + FP}",
        },
    )

    met["balanced_accuracy"] = Expression(
        name="balanced_accuracy",
        symbolic_expr="(sensitivity + specificity) / 2",
        params={"sensitivity": 1, "specificity": 1},
        metadata={
            "category": "metric",
            "subcategory": "classification",
            "description": "Balanced accuracy (average of TPR and TNR)",
            "usage": "Imbalanced datasets",
            "formula_latex": r"\frac{TPR + TNR}{2}",
        },
    )

    met["matthews_corr"] = Expression(
        name="matthews_corr",
        symbolic_expr="(tp * tn - fp * fn) / (sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)) + 1e-8)",
        params={"tp": 1, "tn": 1, "fp": 0, "fn": 0},
        metadata={
            "category": "metric",
            "subcategory": "classification",
            "description": "Matthews Correlation Coefficient (MCC)",
            "usage": "Balanced metric even with imbalanced classes",
            "formula_latex": r"MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}",
        },
    )

    met["log_loss_elem"] = Expression(
        name="log_loss_elem",
        symbolic_expr="-(y * log(p + 1e-15) + (1 - y) * log(1 - p + 1e-15))",
        metadata={
            "category": "metric",
            "subcategory": "classification",
            "description": "Element-wise log loss (binary cross-entropy)",
            "usage": "Probabilistic classification evaluation",
            "formula_latex": r"-[y\ln p + (1-y)\ln(1-p)]",
        },
    )

    met["brier_score_elem"] = Expression(
        name="brier_score_elem",
        symbolic_expr="(p - y)**2",
        metadata={
            "category": "metric",
            "subcategory": "classification",
            "description": "Brier score element (probability calibration)",
            "usage": "Assessing calibration of probabilistic predictions",
            "formula_latex": r"(p_i - y_i)^2",
        },
    )

    # ================================================================
    # Regression Metrics
    # ================================================================

    met["mse_elem"] = Expression(
        name="mse_elem",
        symbolic_expr="(y_pred - y_true)**2",
        metadata={
            "category": "metric",
            "subcategory": "regression",
            "description": "Mean Squared Error element",
            "usage": "Standard regression evaluation",
            "formula_latex": r"(\hat{y}_i - y_i)^2",
        },
    )

    met["mae_elem"] = Expression(
        name="mae_elem",
        symbolic_expr="Abs(y_pred - y_true)",
        metadata={
            "category": "metric",
            "subcategory": "regression",
            "description": "Mean Absolute Error element",
            "usage": "Robust regression evaluation",
            "formula_latex": r"|\hat{y}_i - y_i|",
        },
    )

    met["rmse_formula"] = Expression(
        name="rmse_formula",
        symbolic_expr="sqrt(mse_val)",
        params={"mse_val": 1},
        metadata={
            "category": "metric",
            "subcategory": "regression",
            "description": "Root Mean Squared Error from MSE value",
            "usage": "Regression metric in original units",
            "formula_latex": r"RMSE = \sqrt{MSE}",
        },
    )

    met["r_squared"] = Expression(
        name="r_squared",
        symbolic_expr="1 - ss_res / (ss_tot + 1e-8)",
        params={"ss_res": 0, "ss_tot": 1},
        metadata={
            "category": "metric",
            "subcategory": "regression",
            "description": "R² coefficient of determination",
            "usage": "Proportion of variance explained by model",
            "formula_latex": r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}}",
        },
    )

    met["adjusted_r_squared"] = Expression(
        name="adjusted_r_squared",
        symbolic_expr="1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)",
        params={"r2": 0.9, "n_samples": 100, "n_features": 5},
        metadata={
            "category": "metric",
            "subcategory": "regression",
            "description": "Adjusted R² (penalizes extra features)",
            "usage": "Feature selection, model comparison",
            "formula_latex": r"R^2_{adj} = 1 - (1-R^2)\frac{n-1}{n-p-1}",
        },
    )

    met["mape_elem"] = Expression(
        name="mape_elem",
        symbolic_expr="Abs((y_true - y_pred) / (y_true + 1e-8))",
        metadata={
            "category": "metric",
            "subcategory": "regression",
            "description": "Mean Absolute Percentage Error element",
            "usage": "Relative error metric, forecasting",
            "formula_latex": r"\left|\frac{y_i - \hat{y}_i}{y_i}\right|",
        },
    )

    met["smape_elem"] = Expression(
        name="smape_elem",
        symbolic_expr="Abs(y_pred - y_true) / ((Abs(y_pred) + Abs(y_true)) / 2 + 1e-8)",
        metadata={
            "category": "metric",
            "subcategory": "regression",
            "description": "Symmetric MAPE element",
            "usage": "Balanced percentage error, avoids MAPE asymmetry",
            "formula_latex": r"\frac{|\hat{y}-y|}{(|\hat{y}|+|y|)/2}",
        },
    )

    met["huber_loss_elem"] = Expression(
        name="huber_loss_elem",
        symbolic_expr="Piecewise(((y_pred - y_true)**2 / 2, Abs(y_pred - y_true) <= delta), (delta * (Abs(y_pred - y_true) - delta / 2), True))",
        params={"delta": 1.0},
        metadata={
            "category": "metric",
            "subcategory": "regression",
            "description": "Huber loss element (robust to outliers)",
            "usage": "When data has outliers, DQN training",
            "formula_latex": r"\begin{cases} \frac{(y-\hat{y})^2}{2} & |y-\hat{y}|\le\delta \\ \delta(|y-\hat{y}|-\delta/2) & \text{otherwise} \end{cases}",
        },
    )

    met["explained_variance_ratio"] = Expression(
        name="explained_variance_ratio",
        symbolic_expr="lambda_i / (total_variance + 1e-8)",
        params={"lambda_i": 1, "total_variance": 10},
        metadata={
            "category": "metric",
            "subcategory": "regression",
            "description": "Explained variance ratio (PCA eigenvalue fraction)",
            "usage": "PCA component importance, dimensionality reduction",
            "formula_latex": r"\frac{\lambda_i}{\sum_j \lambda_j}",
        },
    )

    # ================================================================
    # Similarity & Set Metrics
    # ================================================================

    met["jaccard_index"] = Expression(
        name="jaccard_index",
        symbolic_expr="n_intersect / (n_union + 1e-8)",
        params={"n_intersect": 1, "n_union": 1},
        metadata={
            "category": "metric",
            "subcategory": "similarity",
            "description": "Jaccard index (Intersection over Union)",
            "usage": "Object detection (IoU), set similarity",
            "formula_latex": r"J = \frac{|A \cap B|}{|A \cup B|}",
        },
    )

    met["dice_coefficient"] = Expression(
        name="dice_coefficient",
        symbolic_expr="2 * n_intersect / (size_a + size_b + 1e-8)",
        params={"n_intersect": 1, "size_a": 1, "size_b": 1},
        metadata={
            "category": "metric",
            "subcategory": "similarity",
            "description": "Dice coefficient (F1 for sets)",
            "usage": "Image segmentation, medical imaging",
            "formula_latex": r"DSC = \frac{2|A \cap B|}{|A| + |B|}",
        },
    )

    met["cosine_distance_metric"] = Expression(
        name="cosine_distance_metric",
        symbolic_expr="1 - dot_ab / (norm_a * norm_b + 1e-8)",
        params={"dot_ab": 1, "norm_a": 1, "norm_b": 1},
        metadata={
            "category": "metric",
            "subcategory": "similarity",
            "description": "Cosine distance = 1 - cosine similarity",
            "usage": "NLP embedding distance, recommendation",
            "formula_latex": r"d = 1 - \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}",
        },
    )

    # ================================================================
    # Ranking Metrics
    # ================================================================

    met["reciprocal_rank"] = Expression(
        name="reciprocal_rank",
        symbolic_expr="1 / rank",
        params={"rank": 1},
        metadata={
            "category": "metric",
            "subcategory": "ranking",
            "description": "Reciprocal Rank: 1/rank of first relevant result",
            "usage": "Information retrieval, search engine evaluation",
            "formula_latex": r"RR = \frac{1}{\text{rank}}",
        },
    )

    met["dcg_elem"] = Expression(
        name="dcg_elem",
        symbolic_expr="rel / log(pos + 1, 2)",
        params={"rel": 1, "pos": 1},
        metadata={
            "category": "metric",
            "subcategory": "ranking",
            "description": "DCG element: relevance / log₂(position + 1)",
            "usage": "Discounted Cumulative Gain for ranking",
            "formula_latex": r"\frac{rel_i}{\log_2(i+1)}",
        },
    )

    met["ndcg_formula"] = Expression(
        name="ndcg_formula",
        symbolic_expr="dcg / (idcg + 1e-8)",
        params={"dcg": 1, "idcg": 1},
        metadata={
            "category": "metric",
            "subcategory": "ranking",
            "description": "Normalized DCG = DCG / ideal DCG",
            "usage": "Comparing ranked lists, search quality",
            "formula_latex": r"nDCG = \frac{DCG}{IDCG}",
        },
    )

    # ================================================================
    # Information Criteria (Model Selection)
    # ================================================================

    met["aic"] = Expression(
        name="aic",
        symbolic_expr="2 * k - 2 * log_likelihood",
        params={"k": 1, "log_likelihood": 0},
        metadata={
            "category": "metric",
            "subcategory": "model_selection",
            "description": "Akaike Information Criterion",
            "usage": "Model selection, complexity-accuracy tradeoff",
            "formula_latex": r"AIC = 2k - 2\ln(\hat{L})",
        },
    )

    met["bic"] = Expression(
        name="bic",
        symbolic_expr="k * log(n_obs) - 2 * log_likelihood",
        params={"k": 1, "n_obs": 100, "log_likelihood": 0},
        metadata={
            "category": "metric",
            "subcategory": "model_selection",
            "description": "Bayesian Information Criterion",
            "usage": "Model selection (stronger penalty for complexity)",
            "formula_latex": r"BIC = k\ln(n) - 2\ln(\hat{L})",
        },
    )

    return met
