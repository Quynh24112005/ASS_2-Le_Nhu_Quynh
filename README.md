Bài tập lớn 2 – Môn Lập trình Python

Sinh viên thực hiện: Lê Như Quỳnh – B23DCCE081

Giới thiệu
Bài tập này nhằm mục tiêu xây dựng, huấn luyện và đánh giá một mô hình học sâu (deep learning) để phân loại ảnh sử dụng mạng nơ-ron tích chập (CNN) trên tập dữ liệu CIFAR-10. Các bước thực hiện bao gồm: chuẩn bị dữ liệu, thiết kế mô hình, huấn luyện với các kỹ thuật kiểm soát overfitting, trực quan hóa kết quả và đánh giá hiệu suất mô hình.

Toàn bộ báo cáo được biên soạn bằng LaTeX nhằm trình bày khoa học.

Nội dung chính
Chương 1 – Chuẩn bị dữ liệu
Sử dụng thư viện torchvision để tải bộ dữ liệu CIFAR-10.
Tiến hành tiền xử lý ảnh bao gồm: lật ngang ngẫu nhiên, cắt ảnh ngẫu nhiên và chuẩn hóa pixel.
Phân chia tập dữ liệu thành 3 phần: train, validation và test.
Lưu trữ dữ liệu dưới dạng DataLoader để phục vụ huấn luyện.

Chương 2 – Xây dựng mô hình CNN
Xây dựng mạng CNN 3 tầng tích chập đơn giản với các lớp Conv2d, ReLU, MaxPool2d, Flatten, Linear, Dropout.
Sử dụng hàm mất mát CrossEntropyLoss và bộ tối ưu hóa Adam.
Số epoch huấn luyện tối đa là 40, có áp dụng kỹ thuật early stopping để tự động dừng khi không cải thiện.

Chương 3 – Huấn luyện và trực quan hóa
Thực hiện huấn luyện mô hình trên tập train, đánh giá hiệu suất trên tập validation.
Ghi lại log huấn luyện qua từng epoch.
Trực quan hóa quá trình học bằng biểu đồ loss và accuracy theo thời gian (learning curves).
Lưu mô hình tốt nhất theo val_loss.

Chương 4 – Đánh giá mô hình
Đánh giá mô hình đã lưu trên tập test để kiểm tra khả năng tổng quát.
Tính độ chính xác tổng thể (test accuracy ≈ 80.43%).
Vẽ ma trận nhầm lẫn (confusion matrix) để phân tích sai số theo từng lớp.
Tính độ chính xác theo từng lớp (class-wise accuracy).
Đưa ra nhận xét về các lớp dễ gây nhầm lẫn như dog vs cat, và gợi ý cải tiến mô hình.
