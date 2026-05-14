Система аналізу впливу тексту на соціальних мережах з використанням моделей машинного навчання.

## Модельні файли

Розміщені в GitHub Releases:

```text
https://github.com/ssukkami/neural-text-effect-analysis/releases/latest
```

Після завантаження модельні файли потрібно розмістити в папці:

```text
model_assets/
```

Очікувані файли:

```text
confounder_mlp_A_FINAL.pth
confounder_scaler_A_FINAL.pkl
text_effect_roberta_B_FINAL.pth
feature_schema.json
```

## Встановлення

Клонування репозиторію:

```bash
git clone https://github.com/ssukkami/neural-text-effect-analysis.git
cd neural-text-effect-analysis
```

Створення віртуального середовища:

```bash
python -m venv .venv
```

Активація середовища у Windows:

```bash
.venv\Scripts\activate
```

Активація середовища у Linux або macOS:

```bash
source .venv/bin/activate
```

Встановлення залежностей:

```bash
pip install -r requirements.txt
```

## Запуск

Перед запуском потрібно переконатися, що модельні файли розміщені в папці `model_assets/`.

```bash
python app.py
```

## Тестування

```bash
pytest
```

або:

```bash
pytest tests/
```
