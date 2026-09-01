import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
def run_feature_ablation(pre_match, feature_cols, model, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)



    X = pre_match[feature_cols]

    y = pre_match["outcome"]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    groups = {
        "Form": [c for c in feature_cols if "form" in c or "points" in c],
        "Attack_xG": [c for c in feature_cols if "xg" in c or "shot" in c or "goals_for" in c],
        "Defense": [c for c in feature_cols if "against" in c or "pressure" in c],
        "Passing": [c for c in feature_cols if "pass" in c],
        "Context": [c for c in feature_cols if "rest" in c or "home" in c],
    }

    combinations = {
        "Form only": groups["Form"],
        "Form + Attack": groups["Form"] + groups["Attack_xG"],
        "Form + Attack + Defense": groups["Form"] + groups["Attack_xG"] + groups["Defense"],
        "All Features": feature_cols
    }

    results = []

    for name, cols in combinations.items():
        cols = list(set(cols))
        if len(cols) == 0:
            continue

        model.fit(
            X_train[cols],
            y_train
        )

      


        pred = model.predict(
            X_test[cols]
        )


        acc = balanced_accuracy_score(
            y_test,
            pred
        )

        results.append({
            "Feature_Set": name,
            "Number_of_Features": len(cols),
            "Balanced_Accuracy": acc
        })

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        os.path.join(output_dir, "feature_ablation_results.csv"),
        index=False
    )

    plt.figure(figsize=(9,5))
    plt.bar(result_df["Feature_Set"], result_df["Accuracy"])
    plt.xticks(rotation=30)
    plt.ylabel("Balanced Accuracy")
    plt.title("Feature Combination Comparison")
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "feature_ablation.png"),
        dpi=300
    )

    plt.show()

    return result_df
