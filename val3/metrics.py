import pandas as pd

df = pd.read_csv("val3/results.csv")

print("=" * 55)
print("        Результаты валидации модели YOLOv11s")
print("=" * 55)
print(df.to_string(index=False))
print("=" * 55)

best = df[df["class"] != "all"].loc[df[df["class"] != "all"]["mAP50"].idxmax()]
worst = df[df["class"] != "all"].loc[df[df["class"] != "all"]["mAP50"].idxmin()]

print(f"Лучший класс:  {best['class']} (mAP50={best['mAP50']})")
print(f"Худший класс:  {worst['class']} (mAP50={worst['mAP50']})")
