import matplotlib.pyplot as plt

marks = [65, 70, 75, 80, 85, 90, 95, 80]

plt.hist(marks, bins=5)
plt.title("Histogram")
plt.show()