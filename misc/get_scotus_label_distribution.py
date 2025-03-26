import matplotlib.pyplot as plt
from datasets import load_dataset


print("loading SCOTUS dataset...")
dataset = load_dataset("lex_glue", "scotus")

split = "test"
d = {el: 0 for el in set(dataset[split]["label"])}

for l in dataset[split]["label"]:
    d[l] += 1

plt.title(f"SCOUTS label distribution on {split} set")
plt.bar(list(set(dataset[split]["label"])), d.values())
plt.show()