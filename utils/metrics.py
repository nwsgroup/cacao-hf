from evaluate import Metric, MetricInfo
from typing import Dict, List, Optional, Union
import numpy as np
import datasets


# ---------------------------------------------
# Specificity Function
# ---------------------------------------------


class SpecificityMetric(Metric):
    def _info(self) -> MetricInfo:
        return MetricInfo(
            description="Specificity metric for multi-class classification",
            citation="",
            features=datasets.Features({
                "predictions": datasets.Value("int32"),
                "references": datasets.Value("int32"),
            }),
            reference_urls=[]
        )
    
    def _compute(
        self,
        predictions: Union[List[int], np.ndarray],
        references: Union[List[int], np.ndarray],
        normalize: bool = True,
        sample_weight: Optional[List[float]] = None,
        labels: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate specificity for multi-class classification.
        
        Args:
            predictions: Predicted labels
            references: Ground truth labels
            normalize: Whether to return counts or ratios
            sample_weight: Sample weights
            labels: List of unique labels
            **kwargs: Additional arguments
            
        Returns:
            Dict containing the specificity score
        """
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        if isinstance(references, list):
            references = np.array(references)
            
        # Get unique labels if not provided
        if labels is None:
            labels = np.unique(np.concatenate([predictions, references]))
            
        # Calculate specificity for each class
        specificities = []
        
        for label in labels:
            # Create binary masks for current class
            true_negative_mask = (references != label)
            pred_negative_mask = (predictions != label)
            
            # Calculate true negatives and false positives
            tn = np.sum((true_negative_mask) & (pred_negative_mask))
            fp = np.sum((true_negative_mask) & (~pred_negative_mask))
            
            # Calculate specificity for current class
            if tn + fp == 0:
                class_specificity = 0.0
            else:
                class_specificity = tn / (tn + fp)
                
            specificities.append(class_specificity)
            
        # Calculate macro average specificity
        macro_specificity = np.mean(specificities)
        
        return {"specificity": macro_specificity}