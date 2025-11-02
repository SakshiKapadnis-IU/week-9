import pandas as pd
from apputil import GroupEstimate

# Example data
X = pd.DataFrame({
    "loc_country": ["Guatemala", "Mexico", "Guatemala", "Mexico", "Ethiopia"],
    "roast": ["Light", "Medium", "Light", "Dark", "Light"]
})
y = [88, 91, 89, 92, 94]

gm = GroupEstimate(estimate="mean")
gm.fit(X, y)

X_new = [["Guatemala", "Light"], ["Mexico", "Medium"], ["Canada", "Dark"]]
preds = gm.predict(X_new)

print("Predictions:", preds)
