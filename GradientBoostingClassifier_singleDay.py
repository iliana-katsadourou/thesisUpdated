import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time


data = pd.read_csv('result_20210911.csv')
pd.set_option('display.max_rows', None)

num_rows_file1 = len(data)
print(f'Αριθμός δεδομένων πριν την επεξεργασία: {len(data)}')

unique_stop_id = data['stop_id'].nunique()
unique_exit_stop_ids = data['exit_stop_id'].nunique()

print(f'Διακριτές τιμές στη στήλη stop_id: {unique_stop_id}')
print(f'Διακριτές τιμές στη στήλη exit_stop_id: {unique_exit_stop_ids}')

data = data.dropna(subset=['exit_stop_id', 'stop_id'])

# Drop unnecessary columns
data_cleaned = data.drop(columns=['GarNr', 'ValidTalonaId', 'TripCompanyCode', 'stop_name', "geometry", "exit_geometry", "exit_stop_name"])

# Transform date and time to categorical variable with 3 levels (Early, Mid-day and Late)
bins = ['00:00:00', '03:00:00', '11:00:00', '17:00:00', '23:59:59']
labels = ['Late', 'Early', 'Mid-Day', 'Late']
s = pd.to_timedelta(pd.to_datetime(data_cleaned['datetime']).dt.time.astype(str))
data_cleaned['datetime'] = pd.cut(s, bins=pd.to_timedelta(bins), labels=labels, ordered=False)

# Make exit_stop_id the target variable
if 'exit_stop_id' in data_cleaned.columns:
    data_copy = data_cleaned.copy()
    # Assign 'exit_stop_id' as the target variable
    y = data_copy['exit_stop_id']
    data = data_copy.drop(columns=['exit_stop_id'])

# Identify targets with less that 500 ocurrences and remove them from dataset
df = pd.value_counts(y).to_frame().reset_index()
stop_id_counts = data_copy["exit_stop_id"].value_counts()
stop_ids_to_delete = stop_id_counts[stop_id_counts < 500]

df = data_copy[~data_copy['exit_stop_id'].isin(stop_ids_to_delete.index)]

y = df['exit_stop_id']
data = df.drop(columns=['exit_stop_id'])
print("Δεδομενα μετα τον καθαρισμο:", len(data))

unique_stop_id_updated = data['stop_id'].nunique()
unique_exit_stop_ids_updated = df['exit_stop_id'].nunique()

print(f'Διακριτές τιμές στη στήλη stop_id μετα τον καθαρισμο: {unique_stop_id_updated}')
print(f'Διακριτές τιμές στη στήλη exit_stop_id μετα τον καθαρισμο: {unique_exit_stop_ids_updated}')

data["datetime"] = pd.factorize(data["datetime"])[0]
data["route"] = pd.factorize(data["route"])[0]
data["direction"] = pd.factorize(data["direction"])[0]
data["stop_id"] = pd.factorize(data["stop_id"])[0]
y = pd.factorize(y)[0]


x_train, x_test, y_train, y_test=train_test_split(data, y, test_size=0.20,random_state=0)


category_counts_cleaned = df['datetime'].value_counts(normalize=True) * 100

print(f"Ποσοστό των δεδομένων ανά κατηγορία (μετά τον καθαρισμό):")
print(f"Late: {category_counts_cleaned.get('Late', 0):.2f}%")
print(f"Early: {category_counts_cleaned.get('Early', 0):.2f}%")
print(f"Mid-Day: {category_counts_cleaned.get('Mid-Day', 0):.2f}%")

scores = []
times = []
total_time = 0
depths = range(1, 21)

for depth in depths:
    start_time = time.time()
    gbr = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=depth, random_state=2024)
    gbr.fit(x_train, y_train)
    y_pred = gbr.predict(x_test)
    scores.append(accuracy_score(y_test, y_pred))
    total_time += time.time() - start_time  # Update total time
    times.append(total_time)  # Store cumulative time

df_gbr = pd.DataFrame({'Depth': depths, 'Accuracy': scores, 'Cumulative Time (s)': times})
print("Gradient Boosting Results: ")
print(df_gbr)

plt.plot(depths, scores)
plt.xlabel('Value of max_depth for Gradient Boosting Classifier')
plt.ylabel('Testing Accuracy')
plt.show()
