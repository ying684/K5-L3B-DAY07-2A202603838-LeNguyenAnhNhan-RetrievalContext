# Phân tích dev — baseline ban đầu và ma trận agent thật

Phần đầu lưu nhận định trước API; phần cuối là kết quả đo thật sau khi có key mới. Trạng thái cuối: đã chọn H0 trên dev và đánh giá test sau chốt, không chỉnh pipeline bằng test.

Run: `results/runs/20261117_025053_322682_L0_dev_retrieval/`. Cấu hình L0, TF-IDF trên B97, query gốc, return_k=3, budget=2400 ký tự. 17/17 query retrieval thành công. Hit@1=12/15, Evidence Hit@3=13/15; context exact-evidence=13/15. Hai câu thiếu evidence không nằm trong mẫu số retrieval.

Đây là phân tích retrieval/context trên dev; chưa chấm grounded answer, citation hay abstention của agent. Không kết luận Dense/Hybrid tốt hơn trước khi chạy thật.

## d06 — tìm thấy đúng chủ đề nhưng thiếu định nghĩa argmin

Query hỏi ý nghĩa argmin theo theta. Gold ở PDF 82, định nghĩa giá trị theta làm L(theta) nhỏ nhất. Top-3 gồm hai chunk PDF 82 về tối ưu/hàm mất mát và một chunk PDF 96 về argmin theo w. Không có đoạn định nghĩa đầy đủ trong context 1.679 ký tự. Chunk gold đứng **rank 6** trong candidate lexical 10.

Nguyên nhân quan sát được: cutoff top-3 loại đoạn cần thiết. Khác biệt ký hiệu theta/θ và văn bản trích xuất dính chữ có thể làm lexical kém chính xác; đây là giả thuyết chưa được tách bằng ablation. Không thêm từ/nhãn gold vào query hoặc filter để sửa điểm.

H1 thử lấy 6 kết quả và tăng budget lên 4.800 ký tự. Hybrid có thể giúp xếp hạng theo ngữ nghĩa, nhưng cần đo. Khi thay cả cutoff và budget, không thể gán toàn bộ tác động cho một yếu tố riêng.

## d12 — candidate có evidence nhưng ranking và budget gây mất thông tin

Query hỏi số ảnh và số class của ImageNet. Top-3 lần lượt là các chunk PDF 89, 78 và 92. Chunk thứ ba có nhắc cả 1.2M ảnh và 1000 class trong các câu khác với gold quote; kiểm chứng ngữ nghĩa có thể chấp nhận evidence đó nếu nó thật sự được gửi tới model.

Context thực chỉ giữ hai chunk đầu, dài 1.746 ký tự; chunk thứ ba không vừa budget 2.400 nên bị bỏ. Vì vậy context này thực sự thiếu số liệu ImageNet. Chunk chứa trọn gold quote đứng **rank 8** trong candidate lexical 10. Không được đổi một hit ID/đúng trang thành bằng chứng đủ.

H1 thử tăng budget; H2 mở rộng candidate và cho tối đa bốn lượt tìm để model có cơ hội tìm lại theo tên thực thể/điều kiện còn thiếu. Việc model có tìm lại hiệu quả hay không phải xem trace API thật. Không suy từ các self-retrieval smoke bằng vector tài liệu.

## Việc dự kiến bổ sung (ghi trước khi có key mới)

- Chạy D0/L0/H0 và H1/H2 với model thật trên dev; đọc context/answer/citation từng case và ghi nhãn reviewer đúng loại.
- Đo trước/sau cho hai giả thuyết; giữ kết quả giảm điểm hoặc không cải thiện.
- Chọn theo protocol; hiện **chưa có cấu hình được chọn**, chưa mở khóa test.

## Kết quả dev thật sau khi cập nhật key

Đã chạy đủ 85 query-config với `openai/gpt-4.1-mini`, strict tool schema, cùng B97 và model embedding. Tất cả 85 lượt hoàn tất không lỗi hệ thống; nhãn nguồn được Codex đối chiếu từng câu/citation, không phải audit độc lập của con người. Registry: `artifacts/dev_runs.json`; số liệu/nhãn: `artifacts/dev_reviewed_results.json` và từng `reviews.json`.

| Cấu hình | Hit@1 | Hit@3 | Context quote-evidence | Context grade 2 (đọc nội dung) | Grounded mean /2 | Từ chối nhầm | API USD |
|---|---|---|---|---|---|---|---|
| D0 | 10/15 | 13/15 | 11/15 | 11/15 | 0,667 | 8/15 | 0,01627366 |
| L0 | 12/15 | 13/15 | 12/15 | 13/15 | 0,667 | 9/15 | 0,01419640 |
| H0 | 10/15 | 15/15 | 13/15 | 14/15 | 1,000 | 6/15 | 0,01402202 |
| H1 | 10/15 | 15/15 | 14/15 | 14/15 | 0,467 | 10/15 | 0,01920333 |
| H2 | 10/15 | 15/15 | 15/15 | 15/15 | 0,600 | 10/15 | 0,01982154 |

H0 tốt nhất theo grounded mean là tiêu chí ưu tiên đã công bố. H1/H2 không được báo là cải thiện thành công: thêm context giúp evidence nhưng không tự sửa lỗi câu trả lời/citation. H2 tăng khả năng giữ evidence cho d11, song output vẫn bị guard từ chối vì thiếu citation trong answer. H1 tăng chi phí khoảng 36,9% và H2 khoảng 41,4% so H0 trong lượt đo này. Nhiều yếu tố cùng đổi, một lượt/config và provider có biến động; không kết luận nhân quả riêng từng tham số.

### Lỗi mới quan trọng từ agent dev

- **D0/d01 và nhiều câu:** model trả nội dung và danh sách citation ID nhưng bỏ marker `[id]` trong trường answer. Guard giữ hợp đồng đầu ra và chuyển thành abstain; chấm answer 0/từ chối nhầm, dù context có đủ evidence. Không thay raw trace bằng kết quả đúng giả. Đây là giới hạn thực của pipeline hiện tại.
- **d11:** nhiều config chỉ giữ đoạn nói patch không cần pixel trùng nhau; thiếu tiêu chí khoảng cách Euclid nhỏ. L0/H0 còn cite mảnh câu PDF 92 không hỗ trợ phát biểu về patch. Có ID trong context không chứng minh citation đúng. D0/H1 cite đoạn nền PDF 90, hỗ trợ phần không cần pixel trùng nhưng vẫn thiếu câu trả lời đủ.
- **d05:** một số answer thêm RL/số triệu ván/chi tiết AlphaGo ban đầu, trong khi citation Zero chỉ nêu không học ván người, tự chơi và 40 ngày. Chấm một phần và citation không hỗ trợ toàn bộ thay vì mặc định đúng do hai ý chính đúng.
- **H1/d06:** có cách diễn đạt đồng nhất argmin với cực tiểu trong khi nguồn phân biệt minimum và cực tiểu; chấm một phần. **D0/d13:** nói learning rate nhỏ nhưng citation không chứa khuyến nghị này, chấm 0 do thiếu evidence chính.

Context grade theo nội dung được lưu riêng với metric quote đã công bố: e2154d0... PDF 92 có cả 1.2M ảnh và 1000 class dù khác nguyên văn gold; vì thế L0/H0 d12 có thể context grade 2 trong khi quote metric không hit. Không sửa gold hoặc dùng test để đổi metric.

Chốt bằng `src.run select` sau phần review này; các bước test chỉ bắt đầu khi selection artifact được tạo và runtime/config/model không đổi.
