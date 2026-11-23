# Project brief: Các kiểu ngày ô nhiễm không khí ở Hà Nội

## Câu hỏi nghiên cứu
**Ở Hà Nội có những kiểu ngày ô nhiễm nào, và mỗi kiểu thường đi kèm với điều kiện gì?**

- Đối tượng chính: nồng độ PM2.5 theo giờ tại Hà Nội.
- Đơn vị phân tích: một ngày (đường cong PM2.5 trong 24 giờ).
- Hướng tiếp cận: data analysis để hiểu dữ liệu, sau đó data mining để tìm các kiểu ngày và điều kiện đi kèm. ML dự báo chỉ là phần mở rộng (tùy chọn).

## Vì sao không dùng trực tiếp dân số, số nhà máy, số xe
Các biến này thường chỉ có số liệu theo năm và theo tỉnh, trong khi PM2.5 thay đổi theo giờ. Trong chuỗi thời gian của một thành phố, chúng gần như là hằng số. Thay vào đó, dự án dùng các biến đại diện (proxy) có cùng độ phân giải:

| Nguồn ô nhiễm | Biến đại diện |
|---|---|
| Giao thông | Giờ trong ngày, ngày thường/cuối tuần, giờ cao điểm, Tết, giãn cách COVID |
| Đốt rơm rạ, đốt mở | Điểm cháy vệ tinh theo ngày, mùa gặt |
| Công nghiệp / ô nhiễm từ vùng khác | Hướng gió, tốc độ gió |
| Khí tượng (khả năng phát tán) | Tốc độ gió, độ ẩm, mưa, nhiệt độ, áp suất, chênh lệch nhiệt độ (gợi ý nghịch nhiệt) |

## Nguồn dữ liệu dự kiến
- **PM2.5 theo giờ:** trạm AirNow của Đại sứ quán Mỹ tại Hà Nội; OpenAQ (tổng hợp nhiều trạm). *Cần kiểm tra khoảng năm có dữ liệu và tỷ lệ thiếu.*
- **Thời tiết theo giờ:** Open-Meteo Historical Weather API (miễn phí, dựa trên dữ liệu tái phân tích).
- **Điểm cháy:** NASA FIRMS (MODIS/VIIRS), lọc theo vùng đồng bằng Bắc Bộ.
- **Lịch:** ngày Tết Nguyên đán từng năm; các đợt giãn cách ở Hà Nội năm 2020 và 2021. *Cần xác minh ngày chính xác.*

## Ngưỡng tham chiếu (cần kiểm tra lại trước khi dùng)
- Hướng dẫn WHO 2021: PM2.5 trung bình 24 giờ ≤ 15 µg/m³.
- Quy chuẩn Việt Nam (QCVN 05:2023/BTNMT): PM2.5 trung bình 24 giờ ≤ 50 µg/m³.

## Kế hoạch theo giai đoạn

### Giai đoạn 1: Thu thập và làm sạch
- Tải PM2.5 và thời tiết, chuẩn hóa múi giờ (Asia/Ho_Chi_Minh), ghép theo giờ.
- Xử lý giá trị thiếu và giá trị âm hoặc bất thường; ghi lại tỷ lệ thiếu theo năm và tháng.
- Đầu ra: một bảng sạch, mỗi dòng là một giờ.

### Giai đoạn 2: Data analysis (khám phá)
- Xu hướng theo năm, mùa, tháng, giờ trong ngày, thứ trong tuần.
- So sánh dịp Tết và giãn cách với ngày thường.
- Tương quan giữa PM2.5 và các biến thời tiết.

### Giai đoạn 3: Data mining
- **Clustering** các ngày theo đường cong PM2.5 24 giờ (chuẩn hóa; thử k-means, hierarchical, có thể DTW). Đặt tên cho từng cụm, ví dụ "đỉnh sáng-tối kiểu giao thông", "cao suốt ngày kiểu nghịch nhiệt", "tăng đột ngột".
- **Mô tả từng cụm:** phân phối thời tiết, mùa, số điểm cháy, ngày đặc biệt.
- **Association rules** (Apriori/FP-Growth) trên các điều kiện đã rời rạc hóa, nhằm vào nhãn cụm hoặc vượt ngưỡng.
- **Anomaly detection:** tìm ngày bất thường và đối chiếu với tin tức.

### Giai đoạn 4 (tùy chọn): Dự báo
- Dự báo PM2.5 ngày hôm sau từ thời tiết và độ trễ (lag).

## Sản phẩm đầu ra
- Notebook có thể chạy lại từ đầu đến cuối.
- Bộ dữ liệu đã làm sạch (nếu giấy phép nguồn cho phép chia sẻ).
- Bài viết ngắn mô tả các kiểu ngày ô nhiễm và điều kiện đi kèm, có biểu đồ, dành cho người không chuyên.

## Lưu ý
- Kết quả là **mối liên hệ**, không phải quan hệ nhân quả; cần viết cẩn trọng khi diễn giải.
- Một trạm không đại diện cho cả thành phố; ghi rõ giới hạn này.
- Chỉ dùng dữ liệu môi trường và dữ liệu tổng hợp, không dùng dữ liệu cá nhân.

## Nhật ký quyết định
- 2026-11-21: Chốt câu hỏi và hướng tiếp cận dùng biến đại diện thay vì số liệu thống kê năm.
- 2026-11-21: Dùng chuỗi PM2.5 từ kho lưu trữ OpenAQ thay cho AirNow, vì AirNow đã ngừng chia sẻ dữ liệu.
- 2026-11-22: Dùng log(PM2.5) khi so sánh, vì phân phối gốc lệch phải (skew 2,08 → −0,27 sau log, sau khi loại lỗi đo gần 0).
- 2026-11-22: So sánh điều kiện đi kèm trong cùng mùa, vì mùa làm nhiễu mối liên hệ (ví dụ nhiệt độ chủ yếu đại diện cho mùa).
- 2026-11-22 (clustering): Tách mỗi ngày thành hình dạng (log PM2.5 trừ trung bình log của ngày, đưa vào k-means) và mức (3 bậc ≤25, 25–50, >50 µg/m³). Chọn k=4 vì ổn định và dễ diễn giải (bootstrap ARI 0,81 sau khi loại lỗi đo; k=3 là 0,87 nhưng kém chi tiết hơn).
- 2026-11-22 (association rules): Rời rạc hóa thời tiết theo tercile trong cùng tháng; mưa theo ngưỡng 1 mm (ETCCDI); hướng gió 3 cung (Đông Bắc, Đông Nam, Tây); cháy nhiều = >0 và ≥ P75 của tháng.
- 2026-11-22 (association rules): Khai phá trong từng mùa là cách chính; min_support = 20 ngày mỗi mùa, max_len = 4.
- 2026-11-22 (association rules): Lọc luật: vế phải là kiểu ngày/mức; lift ≥ 1,3; luật có ích (+0,05 confidence); Fisher + Benjamini–Hochberg q < 0,05.
- 2026-11-22 (association rules): Tiêu chí vững: lift > 1 ở ≥ 75% số năm (≥ 4 năm, ≥ 5 ngày/năm) và lift > 1,1 trên 2021–2025.
- 2026-11-22: Gắn cờ lỗi đo: ngày có ≥ 6 giờ PM2.5 ≤ 3 µg/m³ thì các giờ đó thành NaN (flag_suspect_zero); 26 ngày bị ảnh hưởng, 14 ngày bị loại. Chạy lại notebook 01–06, kết luận chính không đổi.
- 2026-11-23 (sequential mining, notebook 07): Đợt = các ngày liền nhau mức cao (>50 µg/m³). Kết quả: ngày trước đợt là `Tăng` ở 47% số đợt, ngày cao cuối là `Giảm` ở 47% (cả hai lift ~2,5 so với cùng mùa, vững 9–10/10 năm và holdout 2021–2025). Phần giữa đợt không cố định; Markov bậc 2 không tốt hơn bậc 1, tức không có "trí nhớ" quá 1 ngày. Lưu ý: khuôn phần lớn phản ánh việc đợt bắt đầu buổi tối và tan ban ngày (hình dạng tính 0–23 giờ).
