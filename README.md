# Process Model Analyzer

Веб-приложение для идентификации моделей процессов и расчёта PID параметров.

## Возможности

- Загрузка данных CSV / Excel
- Идентификация **FOPDT**, **SOPDT**, **Интегрирующий процесс**
- Автоматический выбор лучшей модели по R²
- Расчёт PID параметров (Desired Response Method)
- Корректировка и визуализация PID регулятора
- Тёмная тема, интерактивные графики Plotly

## Запуск локально

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Деплой на Streamlit Cloud

1. Форкните этот репозиторий
2. Зайдите на https://share.streamlit.io
3. New app → выберите репозиторий → Main file: `app.py`
4. Deploy
