import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Kernel Ridge Regression with different alpha values

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

batch_size = 1000
scores = []
times = []
total_time = 0
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]

for alpha in alphas:
    start_time = time.time()
    model = KernelRidge(alpha=alpha, kernel='rbf')

    for i in range(0, len(x_train_scaled), batch_size):
        x_batch = x_train_scaled[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        model.fit(x_batch, y_batch)

    predictions = model.predict(x_test_scaled)
    y_pred = [1 if p > 0.5 else 0 for p in predictions]

    scores.append(accuracy_score(y_test, y_pred))
    total_time += time.time() - start_time
    times.append(total_time)

df_kr_alpha = pd.DataFrame({'Alpha': alphas, 'Accuracy': scores, 'Cumulative Time (s)': times})
print("Kernel Ridge Regression Results (Alpha values) with RBF and Mini-Batches: ")
print(df_kr_alpha)

plt.plot(alphas, scores)
plt.xscale('log')
plt.xlabel('Value of alpha for Kernel Ridge Regression')
plt.ylabel('Testing Accuracy')
plt.show()


# # Kernel Ridge Regression for different gamma values

batch_size = 1000
scores = []
times = []
total_time = 0
gammas = [0.001, 0.01, 0.1, 1, 10, 100]

for gamma in gammas:
    start_time = time.time()
    model = KernelRidge(gamma=gamma, kernel='rbf')

    for i in range(0, len(x_train_scaled), batch_size):
        x_batch = x_train_scaled[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        model.fit(x_batch, y_batch)

    predictions = model.predict(x_test_scaled)
    y_pred = [1 if p > 0.5 else 0 for p in predictions]

    scores.append(accuracy_score(y_test, y_pred))
    total_time += time.time() - start_time
    times.append(total_time)

df_kr_gamma = pd.DataFrame({'Gamma': gammas, 'Accuracy': scores, 'Cumulative Time (s)': times})
print("Kernel Ridge Regression Results (Gamma values): ")
print(df_kr_gamma)

plt.plot(gammas, scores)
plt.xscale('log')
plt.xlabel('Value of gamma for Kernel Ridge Regression (RBF)')
plt.ylabel('Testing Accuracy')
plt.show()