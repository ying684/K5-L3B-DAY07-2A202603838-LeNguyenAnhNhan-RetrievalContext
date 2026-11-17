# Báo cáo Day 7 — Agentic RAG

**Lê Nguyễn Anh Nhân · 2A202603838 · L3B · 16/11/2026**


## 1. Dữ liệu, model và cách đánh giá

- Nguồn: [Machine Learning cơ bản](https://machinelearningcoban.com/ebook/), Vũ Hữu Tiệp,
  bản 27/03/2018, 400 trang. Cơ bản dùng lại Day 6: **32 trang PDF 78–109, 97 chunk**;
  bonus index cả 400 trang. [Manifest](../../artifacts/data_manifest.json), [DB proof](../../artifacts/db_verification.json).
- Embedding `baai/bge-m3`, 1.024 chiều; PostgreSQL/pgvector cosine, bảng Day 6 chỉ đọc.
  Agent cơ bản dùng `openai/gpt-4.1-mini` qua OpenRouter; bonus dùng `gemini-3.1-flash-lite` qua Google.
- Cơ bản: **17 dev (15 có đáp án +2 thiếu), 7 test (5+2)**. Bonus: **18 dev (15+3), 30 test (24+6)**.
  Gold có nguồn/trang/evidence; câu thiếu có cách kiểm tra vắng evidence. Runtime không nhận gold.
- Hit@1/3 kiểm source/trang/quote trong top-k, chỉ tính câu có đáp án. Context/answer chấm 0/1/2;
  citation phải hỗ trợ nội dung, không chỉ đúng ID. Chỉ chỉnh bằng dev, lưu snapshot trước test;
  lỗi/timeout không được thay bằng mock hoặc loại khỏi mẫu số.
- Có [quyền xử lý API](../../artifacts/permission_record.json), không suy ra quyền phân phối sách.
  Codex hỗ trợ code, câu hỏi và nhãn nguồn; **chưa có audit độc lập của người học**.

## 2. Retrieval và kết quả cơ bản

| ID | Phương pháp | Candidate → tool return | Context ký tự / max search |
|---|---|---:|---:|
| D0 | Dense cosine | 10 → 3 | 2400 / 2 |
| L0 | TF-IDF | 10 → 3 | 2400 / 2 |
| H0 | Dense + lexical RRF | 10 mỗi nhánh → 3 | 2400 / 2 |
| H1 | Hybrid, context rộng | 10 mỗi nhánh → 6 | 4800 / 2 |
| H2 | Hybrid, thêm candidate/lượt | 20 mỗi nhánh → 6 | 4800 / 4 |

RRF dùng tổng `1/(60+rank)`, dedup ID và xếp lại context qua các query; không dùng filter.
Model chọn search, trả lời hoặc abstain; dừng theo quyết định model hoặc budget.

| Split/config | Hit@1 | Hit@3 | Context quote | Answer mean /2 | Median (s) | Usage USD |
|---|---:|---:|---:|---:|---:|---:|
| dev/D0 |10/15|13/15|11/15|0,667|5,391|0,01627366|
| dev/L0 |12/15|13/15|12/15|0,667|3,589|0,01419640|
| dev/H0 |10/15|15/15|13/15|1,000|4,336|0,01402202|
| dev/H1 |10/15|15/15|14/15|0,467|3,721|0,01920333|
| dev/H2 |10/15|15/15|15/15|0,600|3,445|0,01982154|
| test/D0 |4/5|5/5|4/5|0,800|5,696|0,00717382|
| test/H0 |4/5|5/5|4/5|0,800|4,345|0,00635868|

H0 chốt **03:28:01+07**, trước test **03:28:37+07**. Có 85 dev +14 test query-config,
không lỗi hệ thống trong ma trận chính. D0/H0 test đều: context đủ **4/5**, answer 0/1/2 = **3/0/2**,
đúng đủ **2/5**, citation hỗ trợ **2/2**, abstain đúng **2/2**, từ chối nhầm **3/5**.
Citation coverage 2/2 chỉ tính hai câu đã trả lời. Mọi case đều gọi tool; phân bố lượt gọi,
nhãn và lý do từng câu ở [bằng chứng cơ bản](../../artifacts/basic_runs/),
[metrics đầy đủ](../../artifacts/results_summary.json) và [selection](../../artifacts/selection.json).

**Hai lỗi dev và thử cải thiện:** d06 có evidence rank 6 bị top-3 bỏ; d12 có quote rank 8
và đoạn không vừa budget. H1 tăng context, H2 tăng candidate/lượt tìm. H2 đạt context quote
15/15 nhưng answer mean chỉ 0,6/2; chọn H0 vì answer dev tốt hơn. Model cũng thường điền
citations array nhưng thiếu marker trong câu trả lời, bị guard đổi thành abstain.

| Ví dụ H0 test | Kết quả |
|---|---|
| t02 — end-to-end | Đúng: hai khối được huấn luyện cùng nhau; citation chunk `36082bbd2730d183681b587a`, PDF86. |
| t07 — endpoint/giá bge-m3 | Từ chối đúng vì sách không có evidence. |
| t03 — histogram | Context PDF89 đủ nhưng guard `invalid_citations` làm từ chối nhầm. |

Tổng usage cost 7 run cơ bản **0,09704945 USD**; chưa gồm toàn bộ smoke/preflight.
Cache và thứ tự chạy ảnh hưởng time/cost. Chưa chỉnh bằng lỗi test để đo lại.

## 3. Bonus đa phương thức

400/400 trang có native text/block bbox + ảnh JPEG150dpi, **0 lỗi render/hash**;
918.644 ký tự, 8.540 block, 14 trang ít text vẫn giữ ảnh. Không OCR/caption toàn sách;
hình thiếu caption và native math/reading order còn hạn chế. [Audit nguồn](../../artifacts/bonus/source_audit.json).

BM25/reference search → model mở text/ảnh → trả lời có trang + locator hình/bảng/vùng.
B0 chỉ text, B1 thêm ảnh; giới hạn 12step/4search/9trang, tối đa3trang/call, không fallback.
Dev B1 **16/18, 0 lỗi**; hai lượt B0 riêng **12/18 (5 lỗi)** và **14/18 (3 lỗi)**.
B1 chốt **06:15:09+07**, test bắt đầu **06:15:36+07**; không ghép câu đúng giữa các run.

| Nhóm test | Đạt / Tổng | Tỷ lệ |
|---|---:|---:|
| Bảng |6/6|100%|
| Ảnh/sơ đồ |4/5|80%|
| Biểu đồ |3/5|60%|
| Multi-hop |4/4|100%|
| Text |3/4|75%|
| Thiếu evidence |1/6|16,67%|
| **Toàn bộ** |**21/30**|**70% — chưa đủ bonus**|

5 lỗi cuối gồm 2 network/timeout và 3 guard. bt12 thêm số sai dù đọc đúng output;
bt26–bt28 prose nêu thiếu thông tin đúng nhưng status `answered`, fail theo contract bảo thủ.
ADVANCED chấp nhận prose thiếu evidence: kể cả tính thêm ba câu này cũng chỉ **24/30**, dưới 27/30.
Giữ nguyên nhãn 21/30. bt29/bt30 yêu cầu quá số trang/call, không đổi lỗi thành abstain đúng.

Test dùng 108 request, tổng774,95s, median21,32s (có pacing/tool). Gemini không trả billed USD,
nên chi phí thực **chưa xác định**, không ghi 0; [usage/cost](../../artifacts/bonus/cost_report.json).
[30 nhãn/lý do và trace](../../artifacts/bonus/runs/20261117_061536_933635_B1_test/),
[selection](../../artifacts/bonus/selection.json), [lịch sử thử](../../artifacts/bonus/trial_history.json).
V9/V10 chỉ dev; V11 dừng17/18, chưa chấm, không tính thành kết quả test.

## 4. Kết luận và bài nộp

Đã có bằng chứng cho các nhóm RUBRIC cơ bản: dữ liệu/benchmark, ba retrieval, agent/context,
đánh giá, hai cấu hình cải thiện, tái lập/báo cáo. Kết quả thấp được báo đúng, không tự cho điểm.
Theo RULES: giữ lỗi, tách gold, chọn bằng dev, có nguồn/quyền, không công bố secret/sách.
Theo SUBMISSION: code, benchmark, trace/nhãn và báo cáo có tại local; **quyền xem repo của người chấm,
push và gửi link chưa xác minh/thực hiện**. Bonus chưa đạt điều kiện +15.

Cách chạy ở [RUNBOOK](../../docs/RUNBOOK.md); kiểm tra offline ở [verification](../../artifacts/verification.json).
Chỉ còn một `requirements.txt`. Việc gộp dependency đổi fingerprint đóng gói;
snapshot đã dùng đo được giữ trong [frozen runtime](../../artifacts/frozen_runtime.zip),
không sửa hash/nhãn lịch sử hoặc chạy lại model. Raw trace/PDF/index/archive giữ local và được ignore.
