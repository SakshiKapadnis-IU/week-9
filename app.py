import pandas as pd
from apputil import GroupEstimate

# Example data
X = pd.DataFrame({
    "country": ["Guatemala", "Mexico", "Guatemala", "Mexico", "Ethiopia"],
    "roast": ["Light", "Medium", "Light", "Dark", "Light"]
})
y = [88, 91, 89, 92, 94]

# Create and fit model
gm = GroupEstimate(estimate="mean")
gm.fit(X, y)

# Predict on new samples
X_new = [["Guatemala", "Light"], ["Mexico", "Medium"], ["Canada", "Dark"]]
predictions = gm.predict(X_new)

print("Predictions:", predictions)
