# DataMining_TSNE
Đây là repo chứa code chạy TSNE trên một tập dữ liệu khách hàng được thu thập từ một công ty thương mại điện tử có trụ sở tại São Paulo, Brazil, vào quý 4 năm 2024. Công ty này chuyên cung cấp các sản phẩm thuộc nhiều danh mục như điện tử, thời trang, thực phẩm, nội thất, thể thao và làm đẹp. Dữ liệu phản ánh thông tin về hành vi mua sắm của 70.000 khách hàng, bao gồm các yếu tố như thu nhập, tần suất mua hàng, phương thức thanh toán và mức độ trung thành.

Để chạy, cài các thư viện:
pandas → dùng để xử lý dữ liệu dạng bảng
```bash
pip install pandas
```
scikit-learn (thư viện chứa StandardScaler, TSNE, KMeans)
```bash
pip install scikit-learn
```
matplotlib → dùng để vẽ biểu đồ
```bash
pip install matplotlib
```
seaborn → vẽ biểu đồ đẹp hơn, hỗ trợ màu sắc tốt
```bash
pip install seaborn
```

Sau đó chạy file khachhang.py và đợi, tự động biểu đồ TSNE sẽ hiện ra
