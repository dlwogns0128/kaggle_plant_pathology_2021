import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


ROOT_PATH = "/data/kaggle/plant-pathology-2021-fgvc8"


def remove_duplicates():
    train_df = pd.read_csv(os.path.join(ROOT_PATH, "train.csv"), index_col="image")
    train_files = os.listdir(os.path.join(ROOT_PATH, "train_images"))

    with open("./duplicates.csv", "r") as file:
        duplicates = [x.strip().split(",") for x in file.readlines()]

    # label 두 개가 다르다면 -> 합쳐버리자
    # l1: 'rust', l2: 'scab' => lt: 'rust scab'
    for row in duplicates:
        l1, l2 = train_df.loc[row]["labels"]
        tot_label = set(l1.split(" ")).union(set(l2.split(" ")))
        lt = " ".join(list(tot_label))

        train_df.loc[row[0]]["labels"] = lt
        train_df = train_df.drop(row[1:], axis=0)

    # Multi-label Binarize 'healthy scab frog_eye_leaf_spot complex rust powdery_mildew'
    # 'scab' label -> [0, 1, 0, 0, 0, 0]
    train_df["labels"] = [x.split(" ") for x in train_df["labels"]]

    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(train_df["labels"].values)

    new_train_df = pd.DataFrame(columns=mlb.classes_, data=labels)
    new_train_df.insert(0, "image", train_df.index)

    # divide K fold (seed: 2418)
    X, Y = new_train_df["image"].to_numpy(), new_train_df[
        ["healthy", "scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew"]
    ].to_numpy(dtype=np.float32)
    msss = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=2418)
    folds = np.zeros(len(X), dtype=np.int)
    fold = 0
    for train_index, test_index in msss.split(X, Y):
        folds[test_index] = fold
        fold += 1

    new_train_df["fold"] = folds
    new_train_df.to_csv("./mod_train.csv", index=False)


if __name__ == "__main__":
    remove_duplicates()
