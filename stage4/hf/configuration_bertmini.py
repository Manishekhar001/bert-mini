"""Configuration for the bert-mini ASRS report classifier (Hugging Face format).

This file is self-contained on purpose: it is copied next to the weights and loaded with trust_remote_code=True.
"""
from transformers import BertConfig


class BertMiniConfig(BertConfig):
    """A BERT-mini encoder plus attention pooling and one output layer.

    Extra fields describe how the project uses the model:
        pooling               "attn" (learned token weights), "mean" or "cls"
        report_max_len        tokens kept per report (head + tail truncation)
        report_head_tokens    how many of those come from the start of the report
        calibration_temperature, class_bias, confidence_threshold
                              values fitted on validation data (see the model card)
    Names avoid "temperature" because Transformers reserves it for text generation settings.
    """

    model_type = "bertmini"

    def __init__(self, pooling="attn", classifier_dropout=0.1, report_max_len=512, report_head_tokens=384,
                 calibration_temperature=1.0, class_bias=None, confidence_threshold=None, **kwargs):
        self.pooling = pooling
        self.report_max_len = report_max_len
        self.report_head_tokens = report_head_tokens
        self.calibration_temperature = calibration_temperature
        self.class_bias = class_bias
        self.confidence_threshold = confidence_threshold
        super().__init__(classifier_dropout=classifier_dropout, **kwargs)
