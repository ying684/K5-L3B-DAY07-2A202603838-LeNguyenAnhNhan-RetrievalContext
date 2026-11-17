# Cách chạy Day 7

Python3.11+ (đã dùng3.12.14), chạy từ repo root:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.venv/Scripts/python.exe -X utf8 -B -m unittest discover -s tests -v
.venv/Scripts/python.exe -X utf8 -B scripts/verify_submission.py
```

Key/DB URL chỉ điền vào `.env`, không đưa lên Git. Lần chốt hồ sơ chỉ kiểm offline,
không gọi API. [Báo cáo](../report/individual/2A202603838_LeNguyenAnhNhan.md) và
[verification](../artifacts/verification.json) chứa kết quả thực tế/giới hạn.

## Dữ liệu và DB

Nguồn: [Machine Learning cơ bản](https://machinelearningcoban.com/ebook/), bản27/03/2018,
400trang, SHA-256 `f40e8c07dbdba41b1b2e2fbec9e5c7432752fcc288aa706fa33358a84d23b543`.
PDF đặt `data/day6/raw/book_ML_color.pdf`; chỉ tải từ nguồn được phép.
Cơ bản cần snapshot Day6 strategyB:97chunk trên32trang78–109, embedding bge-m3/1024D.
Danh sách/hash file trong [manifest](../artifacts/data_manifest.json): `prepared/chunks_B.jsonl`,
`prepared/pages.jsonl`, `manifest.json`, `provenance/{protocol,dev_selection}.json`,
`prepared/plan.json` và embedding cache tương ứng dưới `data/day6/`.
Repo không kèm PDF/cache; người chấm cần nguồn hợp lệ và snapshot được cấp quyền.

DB hiện có chỉ đọc; không reindex/xóa bảng Day6. Trên máy mới, dùng DB Day7 riêng:

```powershell
docker compose -p day7_snapshot -f docker-compose.day7.yml up -d --wait
$env:DAY7_RESTORE_DATABASE_URL = 'postgresql://day7:day7_local_only@127.0.0.1:55434/day7_lab'
.venv/Scripts/python.exe -X utf8 -B scripts/restore_day6_snapshot.py --dry-run
.venv/Scripts/python.exe -X utf8 -B scripts/restore_day6_snapshot.py
```

Đặt `DATABASE_URL` local về DB mới, chạy `python -X utf8 -B -m src.run prepare` và
`verify-db`. Restore từ chối DB/port Day6 và bảng đã tồn tại. Bonus dùng index local
`data/bonus/index/` gồm toàn bộ text/block/ảnh400trang; máy chưa có index dùng
`python -X utf8 -B -m src.bonus.experiment index`. Không ghi đè index hiện có.

## Runtime và bằng chứng

- Cơ bản: `python -X utf8 -B -m src.run --help`; cấu hình `configs/*.json`.
  Query embedding `baai/bge-m3`, chat `openai/gpt-4.1-mini` qua OpenRouter.
- Bonus: `python -X utf8 -B -m src.bonus.experiment --help`; `configs/bonus/` dùng
  Gemini3.1FlashLite/minimal, `GEMINI_API_KEY`. Không fallback giữa provider/key.
- Chỉ `--allow-api` cùng `ALLOW_EXTERNAL_PROCESSING=true` mới bật API; cần quyền xử lý
  corpus tại đúng endpoint. Không đưa gold/evidence chuẩn vào model.
- `artifacts/basic_runs/` giữ7run cơ bản; `artifacts/bonus/runs/` giữ3dev+1test bonus.
  Source text/message content đã thay bằng hash/length; context IDs, query/rank/tool,
  answer/citation/usage/review vẫn giữ. Khôi phục context từ B97 hoặc PDF hợp lệ;
  raw đầy đủ local tại `results/` không được push.

Gộp requirements chỉ đổi đóng gói và đường dẫn fingerprint, không chạy lại model.
Các selection/manifest lịch sử giữ nguyên. `artifacts/frozen_runtime.zip` chứa đúng
source/config/benchmark/dependency/protocol đã dùng đo, kèm inventory/hash, không chứa PDF/key.
Muốn kiểm tra lại snapshot cũ, giải nén ZIP vào thư mục riêng, cấp dữ liệu local hợp lệ,
cài `requirements.lock.txt` và `requirements-bonus.txt` bên trong ZIP. Không giải nén đè bản hiện tại.
Fingerprint bản đóng gói mới khác bản đã đo; gate test vẫn chặn khi chưa có freeze mới.
Thí nghiệm mới phải dùng dev/protocol/selection riêng và chấm lại; không sửa nhãn hoặc
ghép run cũ, không dùng test đã xem làm đánh giá độc lập.

`.gitignore` chặn raw results, dữ liệu local, secret, cache, backup và artifact mới;
chỉ cho phép các benchmark/evidence đã duyệt. Kiểm `git status --short` trước commit.
Ignore không tự bỏ file đã tracked; verifier kiểm cả danh sách tracked/có thể add.
