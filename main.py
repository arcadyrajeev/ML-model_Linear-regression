import Model
import matplotlib.pyplot as plt

result = Model.model.predict(Model.X_test)

print(Model.X_test)
print(result)

plt.figure(figsize = (10,7))

plt.scatter(Model.X_test, Model.y_test, c= "g", label="Actual Data")

plt.scatter(Model.X_test, result, c= "b", label="Predicted Data")

plt.legend()
plt.show()