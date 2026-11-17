# Code

Code Day 7 gồm loader/contract/config, adapter pgvector chỉ đọc, embedding tương thích Day 6, Dense/TF-IDF/Hybrid RRF, context nhiều lượt, controller tool calling, agent và evaluator tách gold. CLI: `python -X utf8 -m src.run --help`.

Đã kiểm tra corpus/DB/API thật và chạy đủ ma trận dev/test sau chốt. Xem [RUNBOOK](../docs/RUNBOOK.md) và [báo cáo](../report/individual/2A202603838_LeNguyenAnhNhan.md). Test code offline tách khỏi benchmark agent thật; chất lượng test còn hạn chế do context/citation, đã ghi nhãn đầy đủ.

Bonus độc lập trong `src/bonus/`: index text/ảnh toàn bộ PDF, BM25/reference search,
model tools đọc trang và evaluator có khóa test. CLI: `python -X utf8 -B -m src.bonus.experiment --help`.
Xem [RUNBOOK](../docs/RUNBOOK.md); bản V8 đã test21/30 (70%), chưa đủ bonus.
Đã dừng API; source/config/dependency đã dùng đo nằm trong `artifacts/frozen_runtime.zip`.
