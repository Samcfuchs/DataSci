import seaborn as sns
sns.set(style="ticks")

df = sns.load_dataset("iris")
plot = sns.pairplot(df, hue="species")
plot.savefig("iris_pairplot.png")
