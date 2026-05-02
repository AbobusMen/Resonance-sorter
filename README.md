# Resonance-sorter
A package for classifying the behavior of resonant arguments based on a convolutional network (two models available). The models accept 4000-point input datasets with two channels: the cosine and sine of the resonant angle.

Пакет для классификации поведения резонансных углов с помощью свёрточной сети.
## Структура
- `src/` – модули загрузки, оконной нарезки, инференса, визуализации
- `notebook/` – блокноты .ipynb с примерами 
- `models/` – обученные модели (в репозиторий включена лишь одна из двух, т.к. ограничения по размеру)
- `time_series/` - примеры временных рядов углов для подачи в модель (не загружены из-за ограничений по размеру репозитория)
