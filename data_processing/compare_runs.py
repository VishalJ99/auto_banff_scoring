import pandas as pd

df = pd.read_csv("inflammatory_comparison.csv")

# General stats
total_slides = len(df)
more_in_run2 = (df["difference"] > 0).sum()
more_in_run1 = (df["difference"] < 0).sum()
equal = (df["difference"] == 0).sum()

# Difference stats
mean_diff = df["difference"].mean()
max_gain = df.loc[df["difference"].idxmax()]
max_loss = df.loc[df["difference"].idxmin()]

print(f"Total slides compared: {total_slides}")
print(f"Slides with more WBCs in run2: {more_in_run2}")
print(f"Slides with more WBCs in run1: {more_in_run1}")
print(f"Slides with equal WBCs: {equal}")
print(f"Mean difference: {mean_diff:.2f}")
print(f"Largest increase in run2: {max_gain['slide']} (+{max_gain['difference']})")
print(f"Largest decrease in run2: {max_loss['slide']} ({max_loss['difference']})")