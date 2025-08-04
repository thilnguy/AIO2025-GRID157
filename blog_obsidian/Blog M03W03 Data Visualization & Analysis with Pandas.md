
1. Introduction :

# Giới thiệu về Pandas

**Pandas** là một thư viện mã nguồn mở trong Python, được thiết kế đặc biệt cho việc phân tích và xử lý dữ liệu. Nổi bật với tốc độ, sức mạnh, tính linh hoạt và dễ sử dụng, Pandas được xây dựng dựa trên thư viện **NumPy**. Thư viện này rất hiệu quả khi làm việc với dữ liệu dạng bảng, tương tự như các bảng trong SQL hoặc trang tính Excel. Pandas cung cấp nhiều chức năng để làm sạch, phân tích và xây dựng mô hình dữ liệu, giúp bạn khám phá những đặc điểm chính trong các bộ dữ liệu.

## Các cấu trúc dữ liệu chính

Pandas cung cấp hai cấu trúc dữ liệu cơ bản:

* **Series**: Một mảng một chiều có chỉ mục (labeled index), tương tự như một cột đơn trong bộ dữ liệu.
* **DataFrame**: Cấu trúc bảng hai chiều quan trọng và được sử dụng rộng rãi nhất. Nó bao gồm nhiều hàng và cột, trong đó mỗi cột về cơ bản là một **Series**.

## Các chức năng và khả năng xử lý dữ liệu chính

Pandas cung cấp một bộ công cụ mạnh mẽ cho nhiều tác vụ phân tích dữ liệu:

* **Thao tác và Khám phá dữ liệu**: Hỗ trợ đọc và ghi dữ liệu từ các định dạng phổ biến như CSV, Excel, JSON và SQL. Các hàm như `head()`, `info()`, `describe()` và `dtypes` giúp kiểm tra nhanh cấu trúc dữ liệu, thống kê cơ bản và kiểu dữ liệu.

* **Chọn và Lọc dữ liệu**: Cung cấp các phương thức linh hoạt để truy cập dữ liệu dựa trên nhãn (`.loc[]`, `.at[]`) hoặc vị trí số nguyên (`.iloc[]`, `.iat[]`).

* **Chuyển đổi và Tổng hợp dữ liệu**: Cho phép nhóm dữ liệu (`groupby()`) để tính toán các số liệu thống kê tổng hợp, sắp xếp dữ liệu (`sort_values()`) và áp dụng các hàm tùy chỉnh (`apply()`) cho các hàng hoặc cột.

* **Làm sạch dữ liệu**: Có khả năng xử lý linh hoạt các dữ liệu bị thiếu bằng cách điền giá trị (`fillna()`) hoặc loại bỏ chúng (`dropna()`). Nó cũng hỗ trợ sửa đổi định dạng dữ liệu (ví dụ: chuyển đổi sang số, ngày giờ) và loại bỏ các bản ghi trùng lặp (`drop_duplicates()`).

* **Phân tích chuỗi thời gian (Time Series Analysis)**: Tích hợp các kỹ thuật nâng cao như `rolling()` để tính toán số liệu thống kê trên các cửa sổ di động và `resample()` để thay đổi tần suất dữ liệu. Nó cũng bao gồm lập chỉ mục dựa trên thời gian để truy cập dữ liệu theo các ngày và giờ cụ thể.

## Các ứng dụng phổ biến

Pandas được ứng dụng rộng rãi trong nhiều lĩnh vực khác nhau:

* **Khoa học dữ liệu**: Được sử dụng để tiền xử lý dữ liệu, phân tích khám phá dữ liệu (EDA) và trích xuất đặc trưng.

* **Học máy (ML)**: Hỗ trợ chuẩn hóa và chuyển đổi dữ liệu đầu vào cho các mô hình học máy.

* **Phân tích kinh doanh và tài chính**: Được sử dụng để phân tích doanh thu, chi phí, lợi nhuận và phân khúc khách hàng.

* **Xử lý Dữ liệu lớn / Dữ liệu nhật ký (Log Data)**: Giúp làm sạch, lọc và chuyển đổi dữ liệu từ các hệ thống lớn.

* **Trực quan hóa dữ liệu**: Kết hợp với các thư viện khác như Matplotlib và Seaborn để tạo biểu đồ và hình ảnh trực quan cho dữ liệu.

* **Tự động hóa công việc văn phòng**: Tự động đọc, cập nhật và viết báo cáo từ các tệp Excel/CSV, hợp lý hóa các tác vụ văn phòng thường ngày.
1. Pandas introduction
	1. Data loading & Common data type
	2. Data retrieval (index, slicing, cleaning...)
2. Example
	1. Iris dataset
3. Extra
	1. Visualize data with matplotlib/seaborn
	2. Analyze data
4. Conclusion