"""Per-class results and confusion matrix as Markdown. NumPy only, so it runs anywhere."""
from __future__ import annotations

import numpy as np


def confusion_matrix(labels, predictions, n_classes: int) -> np.ndarray:
    """Rows are the true class, columns are the predicted class."""
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(labels, dtype=np.int64), np.asarray(predictions, dtype=np.int64)), 1)
    return matrix


def per_class_rows(matrix: np.ndarray, classes: list[str]) -> list[dict]:
    """Precision, recall, F1 and support for each class, plus the class it is most often confused with."""
    rows = []
    for c, name in enumerate(classes):
        tp = int(matrix[c, c])
        support = int(matrix[c].sum())
        predicted = int(matrix[:, c].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        errors = matrix[c].copy()
        errors[c] = 0
        top = int(errors.argmax()) if errors.any() else None
        rows.append({"class": name, "support": support, "predicted": predicted, "precision": precision,
                     "recall": recall, "f1": f1,
                     "most_confused_with": classes[top] if top is not None else "",
                     "most_confused_count": int(errors[top]) if top is not None else 0})
    return rows


def markdown_per_class(rows: list[dict]) -> str:
    lines = ["| class | test reports | precision | recall | F1 | most often mistaken for |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in sorted(rows, key=lambda item: -item["support"]):
        confused = (f"{row['most_confused_with']} ({row['most_confused_count']})"
                    if row["most_confused_with"] else "")
        lines.append(f"| {row['class']} | {row['support']} | {row['precision']:.2f} | {row['recall']:.2f} | "
                     f"{row['f1']:.2f} | {confused} |")
    return "\n".join(lines)


def markdown_confusion(matrix: np.ndarray, classes: list[str]) -> str:
    """Row-normalised confusion matrix: each row shows what percent of that true class went to each column.

    Columns use the class numbers from the key below the table, so the table stays narrow. Zero cells are blank.
    """
    header = "| true class \\ predicted | " + " | ".join(str(i) for i in range(len(classes))) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in classes) + " |"
    lines = [header, divider]
    for c, name in enumerate(classes):
        total = matrix[c].sum()
        cells = []
        for p in range(len(classes)):
            percent = 100.0 * matrix[c, p] / total if total else 0.0
            cells.append(f"**{percent:.0f}**" if p == c and percent >= 0.5
                         else (f"{percent:.0f}" if percent >= 0.5 else ""))
        lines.append(f"| {c} {name} (n={int(total)}) | " + " | ".join(cells) + " |")
    key = "; ".join(f"{i} = {name}" for i, name in enumerate(classes))
    return "\n".join(lines) + f"\n\nColumn key: {key}."
