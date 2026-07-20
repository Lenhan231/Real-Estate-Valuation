# 🔍 XAI (Explainable AI) Analysis

**Giải thích model prediction - tại sao model dự đoán giá như vậy?**

## 📁 Nội dung

- `01_shap_analysis.ipynb` - SHAP values & feature importance
- `02_partial_dependence.ipynb` - Mối quan hệ feature-price
- `output/` - Generated plots & visualizations

## 🎯 Mục đích

1. **SHAP values** - Giá trị đóng góp của từng feature vào prediction
2. **Feature importance** - Features quan trọng nhất
3. **Partial dependence** - Khi feature thay đổi, giá thay đổi như thế nào
4. **LIME** - Giải thích dự đoán từng instance cụ thể

## 🚀 Cách chạy

```bash
# Chạy SHAP analysis
jupyter notebook 01_shap_analysis.ipynb

# Hoặc chạy Python script
python generate_xai_report.py
```

## 📊 Output

- SHAP summary plots
- Feature importance rankings
- Partial dependence plots
- HTML report cho presentation
