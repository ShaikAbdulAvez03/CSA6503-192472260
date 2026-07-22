import matplotlib.pyplot as plt

labels = ["Python", "Java", "C++"]
sizes = [40, 35, 25]
plt.pie(sizes, labels=labels)
plt.title("Pie Chart")
plt.show()