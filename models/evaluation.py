from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def generate_evaluation_report(y_true, y_pred, y_pred_proba=None, model_name='Model'):
    """
    Generate comprehensive evaluation report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (optional)
        model_name: Name of the model
    
    Returns:
        dict: Evaluation metrics
    """
    try:
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Classification Report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Calculate additional metrics
        metrics = {
            'model_name': model_name,
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            }
        }
        
        # Add AUC if probabilities provided
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            metrics['auc_roc'] = auc(fpr, tpr)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        
        logger.info(f"Evaluation report generated for {model_name}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error generating evaluation report: {str(e)}")
        raise

def plot_confusion_matrix(y_true, y_pred, labels=['Non-Default', 'Default'], 
                         title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix heatmap
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        save_path: Path to save plot (optional)
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")
        raise

def plot_roc_curve(y_true, y_pred_proba, model_name='Model', save_path=None):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        model_name: Name of the model
        save_path: Path to save plot (optional)
    
    Returns:
        tuple: (fpr, tpr, auc_score)
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'{model_name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.close()
        
        return fpr, tpr, roc_auc
        
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {str(e)}")
        raise

def plot_precision_recall_curve(y_true, y_pred_proba, model_name='Model', save_path=None):
    """
    Plot Precision-Recall curve
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        model_name: Name of the model
        save_path: Path to save plot (optional)
    """
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'{model_name} (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting Precision-Recall curve: {str(e)}")
        raise

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Plot feature importance from tree-based models
    
    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to plot
        save_path: Path to save plot (optional)
    """
    try:
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")
        raise

def compare_models(results_dict, save_path=None):
    """
    Compare multiple models visually
    
    Args:
        results_dict: Dictionary of {model_name: metrics_dict}
        save_path: Path to save plot (optional)
    """
    try:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        model_names = list(results_dict.keys())
        
        data = []
        for metric in metrics:
            row = [results_dict[model].get(metric, 0) for model in model_names]
            data.append(row)
        
        df = pd.DataFrame(data, columns=model_names, index=metrics)
        
        plt.figure(figsize=(12, 6))
        df.T.plot(kind='bar', ax=plt.gca())
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        raise

def calculate_threshold_metrics(y_true, y_pred_proba, thresholds=None):
    """
    Calculate metrics at different probability thresholds
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        thresholds: List of thresholds to evaluate
    
    Returns:
        DataFrame: Metrics at each threshold
    """
    try:
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn
            })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        logger.error(f"Error calculating threshold metrics: {str(e)}")
        raise

def find_optimal_threshold(y_true, y_pred_proba, metric='f1_score'):
    """
    Find optimal prediction threshold based on specified metric
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        metric: Metric to optimize ('f1_score', 'precision', 'recall')
    
    Returns:
        float: Optimal threshold
    """
    try:
        threshold_metrics = calculate_threshold_metrics(y_true, y_pred_proba)
        optimal_idx = threshold_metrics[metric].idxmax()
        optimal_threshold = threshold_metrics.loc[optimal_idx, 'threshold']
        
        logger.info(f"Optimal threshold for {metric}: {optimal_threshold:.3f}")
        return optimal_threshold
        
    except Exception as e:
        logger.error(f"Error finding optimal threshold: {str(e)}")
        raise

def generate_full_evaluation(model, X_test, y_test, feature_names, 
                            model_name='Model', output_dir='outputs/'):
    """
    Generate complete evaluation report with all plots
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        model_name: Name of the model
        output_dir: Directory to save outputs
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Generate metrics
        metrics = generate_evaluation_report(y_test, y_pred, y_pred_proba, model_name)
        
        # Create plots
        plot_confusion_matrix(y_test, y_pred, 
                            title=f'{model_name} - Confusion Matrix',
                            save_path=f'{output_dir}/{model_name}_confusion_matrix.png')
        
        plot_roc_curve(y_test, y_pred_proba, model_name,
                      save_path=f'{output_dir}/{model_name}_roc_curve.png')
        
        plot_precision_recall_curve(y_test, y_pred_proba, model_name,
                                   save_path=f'{output_dir}/{model_name}_pr_curve.png')
        
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model, feature_names,
                                  save_path=f'{output_dir}/{model_name}_feature_importance.png')
        
        logger.info(f"Full evaluation completed for {model_name}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error in full evaluation: {str(e)}")
        raise