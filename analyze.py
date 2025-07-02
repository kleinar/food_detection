import os
import matplotlib.pyplot as plt
from collections import defaultdict

class_names = {
    0: "salad_with_halloumi", 1: "quesadilla", 2: "borsch",
    3: "greek_salad", 4: "lamb_ribs", 5: "tea_infuser",
    6: "empty_cups", 7: "empty_plate", 8: "tea_cup",
    9: "pumpkin_cream_soup"
}

splits = ["train", "val", "test"]
label_base = "dataset/labels"

os.makedirs("graphs", exist_ok=True)

for split in splits:
    counts = defaultdict(int)
    label_dir = os.path.join(label_base, split)

    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            with open(os.path.join(label_dir, file)) as f:
                for line in f:
                    if line.strip():
                        cls_id = int(line.strip().split()[0])
                        counts[cls_id] += 1

    class_ids = sorted(class_names.keys())
    values = [counts.get(cid, 0) for cid in class_ids]
    labels = [class_names[cid] for cid in class_ids]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Количество объектов")
    plt.title(f"{split.upper()} — распределение классов")
    plt.tight_layout()
    plt.savefig(f"graphs/{split}_distribution.png")
    plt.close()