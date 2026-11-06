# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603801
- **Họ và tên:** Nguyễn Xuân Khánh Linh
- **SEED** (in ở ô đầu notebook): 7307

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra (15/15 bài kiểm tra đạt) | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ (sẽ upload sau khi chụp lại) |

## 2. Kết quả chính

Bảng so sánh cuối Phần 5. Các hàng LLM chỉ chạy trên tập SUB (40 tin), nên cột "F1 (test)" của hàng LLM chính là F1 trên SUB. LLM dùng Groq (`gpt-oss-20b` là model nhỏ, `gpt-oss-120b` là model lớn); model Groq không có trong bảng giá của lab nên cột USD hiện là 0.

| Cách tiếp cận | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.841 | 0.889 | 0.082 | 0 |
| 2. BoW + MLP | 0.918 | 0.974 | 0.372 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.895 | 1.000 | 110.7 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.944 | 1.000 | 39.6 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.954 | 1.000 | 41.1 | 0 |
| 4a. LLM nhỏ zero-shot | 0.889 | 0.889 | 2514.7 | 0 (chưa có giá) |
| 4b. LLM lớn zero-shot | 0.976 | 0.976 | 2540.0 | 0 (chưa có giá) |
| 4c. LLM nhỏ few-shot | 0.905 | 0.905 | 2776.6 | 0 (chưa có giá) |

Nhận xét ngắn: nhóm encoder cho F1 cao nhất trong các cách không dùng LLM và vẫn dưới 50 ms (3b, 3c); BoW nhanh nhất nhưng recall thấp (LogReg: precision 0.987, recall 0.733); LLM chậm hơn hàng chục đến hàng nghìn lần. Đường Pareto (1 − F1 trên SUB so với p95) gồm BoW + LogReg, BoW + MLP và Encoder + LogReg.

## 3. Dự đoán và thực tế

1. **`ml_lr` (learning rate).** Mình đoán `lr=0.001` thì loss giảm rất chậm (đúng), còn `lr=50` thì loss dao động và không hội tụ. Thực tế `lr=50` chỉ vọt lên một lần lúc đầu (khoảng 2.5) rồi giảm về gần 0 (loss cuối 0.004). Vì sao lệch: với logistic regression, sai lệch `p − y` luôn nằm trong khoảng −1 đến 1 nên bước cập nhật bị chặn; mô hình nhảy quá đà rồi tự quay lại, chứ không phân kỳ.
2. **`llm_models` (model lớn so với nhỏ).** Mình đoán model lớn hơn nhưng chênh ít vì bài dễ, và LLM bằng Encoder + LogReg (3b). Thực tế model lớn F1 = 0.976 so với 0.889 của model nhỏ (chênh khoảng 0.09, nhiều hơn dự đoán), và 3b đạt 1.000 trên SUB nên nhỉnh hơn cả model lớn. Vì sao lệch: LLM nhỏ gắn nhãn spam nhầm cho một số tin ham lộn xộn (precision 0.80), mà SUB chỉ có 40 tin nên vài tin sai làm F1 đổi nhiều.
3. **`hard_eval` (case khó).** Mình đoán BoW khó nhất ở nhóm né từ khóa và LLM khó nhất ở nhóm prompt injection. Thực tế BoW + LogReg khó nhất ở spam giọng tự nhiên (đúng 0%), rồi thao túng/lừa đảo (12%) và né từ khóa (14%); LLM nhỏ khó nhất ở tin tưởng spam nhưng không phải (73%) và nói về spam (80%), còn prompt injection đúng 86% (4a) và 100% (4d). Vì sao lệch: mình đánh giá thấp việc BoW không hiểu giọng văn tự nhiên, và đánh giá quá cao mức LLM bị prompt injection lừa.

## 4. Phân tích case khó

Bộ case khó gồm 50 case gốc và 6 case mình tự thêm (56 case). Model tụt nhiều nhất là **BoW + LogReg**: F1 từ 0.889 (SUB) xuống 0.293, giảm khoảng 0.60 (BoW + MLP giảm 0.506; Encoder giảm 0.31–0.39; hai LLM không tụt, còn tăng nhẹ lên 0.925 và 0.941). Nhóm khó nhất với BoW là spam giọng tự nhiên (đúng 0%).

- **Case #11** (spam giọng tự nhiên): "Hey it's Jess from the gym 😊 we're doing a members-only thing this week, 3 months for the price of 1, just reply JOIN and I'll sort it". BoW sai vì tin dùng toàn từ thân mật, không có từ khóa spam quen thuộc; BoW chỉ cộng điểm theo từng từ rời rạc nên không thấy ý quảng cáo.
- **Case #2** (tin ham giống spam): "FREE pizza in the office kitchen right now, come quick before it's gone". LLM nhỏ có prompt phòng thủ (4d) vẫn đoán spam vì tin có "FREE", "right now", giục giã; model quá cảnh giác nên gắn spam cho tin ham có từ khóa quảng cáo, dù ngữ cảnh là đồng nghiệp rủ ăn.

**Case do mình thêm (bonus).** Hai LLM đúng cả 6/6. BoW + LogReg và BoW + MLP đúng 2/6, Encoder + kNN 1/6, Encoder + LogReg 3/6, Encoder + MLP head 2/6. Case làm sai nhiều model nhất là: tin ham có "prize", "free" (bán bánh nội bộ), spam tiếng Việt không dấu, và tin ham nhắc tới tin trúng thưởng để cảnh báo. Lý do: BoW và embedding bị kéo về cụm spam bởi từ khóa; từ điển BoW là tiếng Anh và encoder MiniLM học chủ yếu tiếng Anh nên không xử lý được tiếng Việt; LLM đọc được ngữ cảnh và nhiều ngôn ngữ.

## 5. Thử thách cá nhân

Đề #6 (SEED 7307, 7307 mod 6 = 5): sửa prompt để LLM trả JSON `{"label": ..., "reason": ...}`, viết hàm parse JSON, chạy lại trên 10 tin SUB, so token và chi phí với bản 1 từ, kết luận có đáng không.

Cách làm: viết `build_json_prompt` (yêu cầu chỉ trả một đối tượng JSON), `parse_json_label` (cắt lấy đoạn từ dấu `{` đầu tiên đến dấu `}` cuối cùng, `json.loads`, trả `None` nếu lỗi rồi đếm là tin không hợp lệ) và `run_challenge` (chạy 10 tin đầu của SUB với model nhỏ, ước lượng token bằng số từ chia 0.75, tính tiền theo giá `gpt-6-luna`).

Kết quả:

| | accuracy | tin không hợp lệ | token vào | token ra | USD / 1k tin | p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 từ | 0.8 | 0 | 637 | 13 | 0.00704 | 286 ms |
| JSON | 0.8 | 0 | 811 | 119 | 0.01404 | 358 ms |

JSON tốn gấp 1.43 lần tổng token và gấp 1.99 lần tiền, trong khi accuracy không đổi và không có tin nào parse lỗi. Tiền tăng nhiều hơn token vì token ra đắt hơn token vào 5 lần và tăng khoảng 9 lần (13 lên 119). **Đáng không?** Với phân loại hàng loạt thì không đáng vì chất lượng không tăng mà chi phí và độ trễ gần gấp đôi; JSON chỉ đáng khi cần con người kiểm tra lại (lý do cho biết model sai vì sao), nên dùng cho tin lưỡng lự.

## 6. Phương án cho sản phẩm thật

Yêu cầu: p95 dưới 50 ms/tin cho 10 triệu SMS/ngày (khoảng 116 tin/giây).

- **Đạt độ trễ:** BoW + LogReg (0.08 ms), BoW + MLP (0.37 ms), Encoder + LogReg (39.6 ms) và Encoder + MLP head (41.1 ms). kNN (110.7 ms) và các LLM (2.5–2.8 giây) không đạt.
- **Chọn:** Encoder + MLP head (F1 test 0.954, SUB 1.000, p95 41.1 ms) hoặc Encoder + LogReg (F1 0.944, p95 39.6 ms, nằm trên đường Pareto). Đánh đổi: phải chạy encoder cho mỗi tin; trên một máy CPU của Colab chỉ đạt khoảng 27 tin/giây, thấp hơn 116 tin/giây cần thiết, nên phải chạy nhiều máy hoặc dùng GPU. BoW + MLP (F1 0.918) là phương án dự phòng rẻ hơn rất nhiều với hàng chục nghìn tin/giây.
- **Chi phí nếu dùng LLM:** với 100 token vào và 1 token ra mỗi tin, giá `gpt-6-luna` ($0.10 / $0.50 mỗi 1M token) là $10.50 cho 1 triệu tin, tức $105/ngày và khoảng $3,150/tháng cho 10 triệu tin/ngày; model lớn (`gpt-6.1-terra`) đắt gấp khoảng 10 lần. LLM cũng chậm hơn nhiều so với SLA 50 ms.
- **Thiết kế lai (router):** dùng Encoder + MLP head cho mọi tin; chỉ tin có xác suất spam p trong [0.3, 0.7] mới chuyển sang LLM. Cách này giữ chất lượng gần LLM nhưng chỉ một phần nhỏ tin phải trả tiền và chờ lâu.

## 7. AI log và điều đã học

**AI log** (các bước dùng AI, đúng như đã làm):

| Phần | Hỏi AI gì | AI làm gì | Mình sửa / kiểm chứng |
| --- | --- | --- | --- |
| Phần 1–4 (TODO 1.1 → 4.4) | Nhờ viết code các hàm TODO, vì lab coach cho phép | AI viết code và chạy thử; với TODO 1.1 (mình tự viết trước) AI chỉ thêm `return X` và `dtype=np.float32` | Chạy các ô kiểm tra, đối chiếu kết quả |
| Các ô dự đoán | Hỏi giải thích khái niệm (F1, learning rate, ReLU, embedding, Pareto) | AI giải thích, đưa quy tắc suy luận và vài câu ví dụ | Mình chọn, chỉnh và viết lại câu dự đoán |
| Giải thích, Câu hỏi, Bắt lỗi, Phản ánh | Nhờ giải thích khái niệm và soạn bản nháp câu trả lời | AI soạn bản nháp dựa trên số liệu mình chạy | Đối chiếu số liệu với kết quả của mình |
| `MY_CASES` (bonus) | Nhờ soạn 6 case khó | AI viết 6 case kèm lý do đánh lừa, rồi đọc kết quả sau khi chạy để viết nhận xét | Mình chạy lại ô stress test và đối chiếu từng case |
| Tính tay | Nhờ làm 5 phép tính | AI đưa lời giải mẫu từng bước và đáp số | Mình viết tay lại các bước trên giấy và đối chiếu đáp số |
| Thử thách #6 | Nhờ viết code theo đề (đã báo lab coach cho phép) | AI viết code và chạy thử bằng API giả lập, rồi chạy trên Colab và viết nháp phần giải thích từ số liệu thật | Mình đối chiếu số liệu với bảng kết quả trong notebook |
| Chạy notebook | Nhờ chạy notebook (Run all) và các ô cần chạy lại | AI bấm chạy trên Colab của mình, cập nhật số liệu trong các câu trả lời cho khớp lần chạy cuối | Mình kiểm tra 15/15 bài kiểm tra đạt trong `lab1_report.json` |
| Side lab bonus | Nhờ viết `solution.py` | AI viết code, chạy 19/19 test với API giả | Chạy pytest |

**Điều mình hiểu ra hôm nay:** accuracy cao không có nghĩa mô hình tốt khi dữ liệu lệch lớp, nên phải xem precision, recall và F1; bộ test cùng phân phối với tập train cho điểm quá lạc quan, bộ case khó mới lộ điểm yếu thật (BoW không hiểu giọng văn tự nhiên, không xử lý được tiếng Việt); và chọn mô hình cho sản phẩm thật là bài toán đánh đổi giữa chất lượng, độ trễ và chi phí chứ không phải cứ mô hình lớn nhất là tốt nhất.
