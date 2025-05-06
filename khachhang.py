import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "customer_São_Paulo_2024.csv"
numerical_features_for_tsne = ['Income', 'SpendingScore', 'PurchaseFrequency',
                               'LoyaltyScore', 'HouseholdSize', 'CreditScore']

# --- 1. Đọc Dữ liệu ---
try:
    df = pd.read_csv(file_path)
    print("Đã đọc dữ liệu thành công.")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn {file_path}")
    exit()

# --- 2. Chuẩn bị Dữ liệu cho T-SNE ---
# Loại bỏ các hàng có giá trị thiếu trong các cột sẽ dùng cho T-SNE
df_cleaned = df.dropna(subset=numerical_features_for_tsne).copy()
print(f"Số lượng hàng sau khi làm sạch dữ liệu thiếu: {len(df_cleaned)}")

# Chọn dữ liệu cho T-SNE
X_tsne_input = df_cleaned[numerical_features_for_tsne]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_tsne_scaled = scaler.fit_transform(X_tsne_input)
print("Đã chuẩn hóa dữ liệu cho T-SNE.")

# --- 3. Áp dụng T-SNE ---
# n_components=2 để giảm xuống còn 2 chiều
# random_state để kết quả ổn định hơn
# n_iter > 250 (thường là 300-1000) cho kết quả tốt hơn, mặc định là 1000
# learning_rate (mặc định 'auto' hoặc 200)
tsne = TSNE(n_components=2, random_state=42, n_iter=300, learning_rate='auto')
X_tsne_results = tsne.fit_transform(X_tsne_scaled)
print("Đã áp dụng T-SNE.")

df_tsne_plot = pd.DataFrame(data=X_tsne_results, columns=['TSNE-1', 'TSNE-2'], index=df_cleaned.index) # Giữ index để dễ dàng nối lại
df_tsne_plot['SpendingScore'] = df_cleaned['SpendingScore']
df_tsne_plot['LoyaltyScore'] = df_cleaned['LoyaltyScore']

plt.figure(figsize=(14, 6))

# Biểu đồ T-SNE được tô màu theo SpendingScore
plt.subplot(1, 2, 1)
sns.scatterplot(x='TSNE-1', y='TSNE-2', hue='SpendingScore', data=df_tsne_plot, palette='viridis', alpha=0.6)
plt.title('Kết quả T-SNE (Tô màu theo Điểm Chi tiêu)')
plt.xlabel('Thành phần T-SNE 1')
plt.ylabel('Thành phần T-SNE 2')

# Biểu đồ T-SNE được tô màu theo LoyaltyScore
plt.subplot(1, 2, 2)
sns.scatterplot(x='TSNE-1', y='TSNE-2', hue='LoyaltyScore', data=df_tsne_plot, palette='viridis', alpha=0.6)
plt.title('Kết quả T-SNE (Tô màu theo Điểm Trung thành)')
plt.xlabel('Thành phần T-SNE 1')
plt.ylabel('Thành phần T-SNE 2')

plt.tight_layout()
plt.show()

# --- 5. Tìm số cụm tối ưu bằng Phương pháp Elbow ---
print("\n--- Tìm số cụm tối ưu cho K-Means (Phương pháp Elbow) ---")
X_for_clustering = df_tsne_plot[['TSNE-1', 'TSNE-2']] # Dữ liệu T-SNE 2D để phân cụm

inertia = []
range_of_clusters = range(1, 16)

for k in range_of_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_for_clustering)
    inertia.append(kmeans.inertia_)

# Vẽ biểu đồ Elbow
plt.figure(figsize=(8, 4))
plt.plot(range_of_clusters, inertia, marker='o')
plt.title('Phương pháp Elbow để tìm K tối ưu')
plt.xlabel('Số lượng cụm (K)')
plt.ylabel('Inertia (Tổng bình phương khoảng cách)')
plt.xticks(range_of_clusters)
plt.grid(True)
plt.show()

# --- 6. Áp dụng K-Means với số cụm đã chọn ---
optimal_k = input("Nhập số cụm K bạn đã chọn từ biểu đồ Elbow: ")
try:
    optimal_k = int(optimal_k)
    if optimal_k < 1:
         raise ValueError
except ValueError:
    print("Nhập không hợp lệ. Vui lòng nhập một số nguyên dương.")
    exit()


print(f"\n--- Áp dụng K-Means với K = {optimal_k} ---")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_for_clustering)
print("Đã áp dụng K-Means.")

# Thêm nhãn cụm vào DataFrame đã làm sạch ban đầu bằng cách sử dụng index
df_cleaned['Cluster'] = clusters
print("Đã thêm nhãn cụm vào dữ liệu.")

df_tsne_plot['Cluster'] = df_cleaned['Cluster']
print("Đã thêm nhãn cụm vào DataFrame vẽ biểu đồ T-SNE.")

# --- 7. Trực quan hóa kết quả T-SNE với màu theo Cụm ---
plt.figure(figsize=(10, 8))
sns.scatterplot(x='TSNE-1', y='TSNE-2', hue='Cluster', data=df_tsne_plot,
                palette='viridis', alpha=0.6, legend='full') # Sử dụng df_tsne_plot để vẽ
plt.title(f'Kết quả T-SNE với {optimal_k} Cụm (K-Means)')
plt.xlabel('Thành phần T-SNE 1')
plt.ylabel('Thành phần T-SNE 2')
plt.show()
print("Đã hiển thị biểu đồ T-SNE với các cụm được tô màu.")


# --- 8. Phân tích đặc điểm của từng cụm ---
print(f"\n--- Phân tích đặc điểm của {optimal_k} Cụm Khách hàng ---")

# Phân tích các đặc điểm số trung bình của từng cụm
print("\nĐặc điểm TRUNG BÌNH của từng cụm:")
print(df_cleaned.groupby('Cluster')[numerical_features_for_tsne].mean())
