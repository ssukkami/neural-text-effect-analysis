# NEURO_INFLUENCE - Diploma Project

Система аналізу впливу тексту на соціальних мережах з використанням моделей машинного навчання.

## 🎯 Основні компоненти

- **Text Analysis:** RoBERTa-base для аналізу тексту
- **Confounder Detection:** MLP для обробки контекстних даних
- **Interactive UI:** Веб-інтерфейс з аналізом A/B та обробкою датасетів

## 📦 Встановлення

### 1. Клонування репозиторію
```bash
git clone https://github.com/YOUR_USERNAME/neuro_influence.git
cd neuro_influence
```

### 2. Встановлення залежностей
```bash
pip install -r requirements.txt
```

### 3. Завантаження моделей

Моделі зберігаються окремо через Git LFS для оптимізації розміру репозиторію.

**Варіант A: Git LFS (рекомендується)**
```bash
# Встановити Git LFS
git lfs install

# Моделі автоматично завантажиться при клонуванні
git clone https://github.com/YOUR_USERNAME/neuro_influence.git
```

**Варіант B: Автоматичне завантаження**
```bash
# Запустити скрипт завантаження моделей
python download_models.py
```

## 🚀 Запуск

### Web Interface
```bash
python app.py
```
Додаток буде доступний за адресою: `http://localhost:5000`

### Тести
```bash
pytest tests/
```

## 📁 Структура проєкту

```
neuro_influence/
├── app.py                      # Flask сервер
├── inference.py                # Логіка передбачень
├── requirements.txt            # Залежності Python
├── download_models.py          # Скрипт завантаження моделей
├── model_assets/
│   ├── feature_schema.json     # Конфігурація ознак
│   ├── confounder_mlp_A_FINAL.pth
│   ├── confounder_scaler_A_FINAL.pkl
│   └── text_effect_roberta_B_FINAL.pth (>100MB, Git LFS)
├── static/
│   ├── app.js                  # Логіка фронтенду
│   └── style.css               # Стилі
├── templates/
│   └── index.html              # Основна сторінка
└── tests/
    ├── conftest.py
    └── test_information_system.py
```

## 🔧 Архітектура

### Backend (Flask)
- Обробка текстових вводів
- Обробка CSV датасетів
- API endpoints для аналізу

### Frontend (HTML/CSS/JS)
- **Single Text:** Аналіз окремого тексту
- **A/B Compare:** Порівняння двох текстів
- **Dataset:** Обробка великих датасетів (до 1200 рядків)

### ML Models
- **TextEffectRoBERTa:** Аналіз впливу тексту за допомогою трансформерів
- **ConfounderMLP:** Коригування результатів за контекстні змінні

## 📊 Функціональність

### 1. Аналіз тексту
```python
predictor = Predictor()
result = predictor.predict_single(text, followers=100, following=50)
```

### 2. Обробка датасетів
Завантажуйте CSV файли з 1200+ рядків для пакетної обробки

## 🛠️ Модельні активи

**Увага:** Файли моделей керуються Git LFS. Якщо Git LFS не встановлено:

```bash
# Встановити Git LFS
# На Windows: https://git-lfs.github.com/
# На macOS: brew install git-lfs
# На Linux: https://github.com/git-lfs/git-lfs/wiki/Installation

git lfs install
git lfs pull
```

Або скористайтесь скриптом `download_models.py`.

## 📋 Вимоги

- Python 3.8+
- PyTorch 2.0+
- transformers
- Flask
- pandas, numpy, scikit-learn

Див. `requirements.txt` для повного списку.

## 🧪 Тестування

```bash
# Запустити всі тести
pytest tests/

# З деталями покриття
pytest tests/ --cov
```

## 📝 Ліцензія

MIT License

## 👤 Автор

Розроблено як дипломний проект.

## 🔗 Посилання

- [Приклад 1: NoSmoking](https://github.com/Xakep777/NoSmoking)
- [Приклад 2: NLP-Relevance-System](https://github.com/da-hardysh/NLP-Relevance-System)

