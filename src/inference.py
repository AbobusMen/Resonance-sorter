import numpy as np
from tensorflow import keras
from pathlib import Path


# Словарь для выбора модели по имени или номеру
MODEL_OPTIONS = {
  '1': 'Model_for_original_data.keras',
  'original': 'Model_for_original_data.keras',
  '2': 'Model_for_undersampling_data.keras',
  'undersampling': 'Model_for_undersampling_data.keras',
}

#Проблема с кастомным слоем аугментации
from .augmentation_layers import RandomReverse, RandomRotation, RandomAngleNoise
DEFAULT_CUSTOM_OBJECTS = {
  'RandomReverse': RandomReverse,
  'RandomRotation': RandomRotation,
  'RandomAngleNoise': RandomAngleNoise,
}


def resolve_model_path(model_id: str | int, models_dir: str | Path = 'models') -> str:
  """
  По идентификатору модели возвращает полный путь к файлу .keras.

  Параметры:
  model_id : str or int
    '1', 'original' – модель для оригинальных данных,
    '2', 'undersampling' – модель для сбалансированных данных.
    Также можно передать сразу путь к конкретному файлу.
  models_dir : str or Path
    Папка, в которой лежат .keras файлы.

  Возврат:
  filepath : str
    Абсолютный или относительный путь к файлу модели.
  """
  models_dir = Path(models_dir)
  if isinstance(model_id, int):
      model_id = str(model_id)

  # Если передан прямой путь к существующему файлу
  if model_id.endswith('.keras') and Path(model_id).exists():
      return model_id

  # Поиск среди предопределённых имён
  filename = MODEL_OPTIONS.get(model_id.lower())
  if filename is None:
      raise ValueError(
          f"Неизвестный идентификатор модели: {model_id}. "
          f"Допустимые варианты: {list(MODEL_OPTIONS.keys())}"
      )
  return str(models_dir / filename)


def load_model(
    model_id: str | int,
    models_dir: str | Path = 'models',
    custom_objects: dict | None = None,
) -> keras.Model:
    """
    Загружает модель без компиляции.
    Параметры:
        custom_objects: словарь с дополнительными классами.
                        Если None, используется DEFAULT_CUSTOM_OBJECTS.
                        На случай добавление моделей без кастомных аугментаций
    """
    path = resolve_model_path(model_id, models_dir)
    # Если custom_objects не передан явно, берём стандартный набор для двух моделей
    if custom_objects is None:
        custom_objects = DEFAULT_CUSTOM_OBJECTS
    model = keras.models.load_model(path, custom_objects=custom_objects, compile=False)
    return model


def predict_on_windows(angle_windows: np.ndarray, model: keras.Model) -> np.ndarray:
  """
  Прогоняет окна одного угла через модель и возвращает вероятности классов.

  Параметры:
  angle_windows : np.ndarray, shape (n_windows, 4000, 2)
    Массив окон для одного угла.
  model : keras.Model
    Загруженная модель.

  Возврат:
  proba : np.ndarray, shape (n_windows, 3)
    Вероятности трёх классов для каждого окна.
  """
  if angle_windows.ndim != 3 or angle_windows.shape[1] != 4000 or angle_windows.shape[2] != 2:
      raise ValueError(
          f"Ожидалась форма (n_windows, 4000, 2), получена {angle_windows.shape}"
      )
  proba = model.predict(angle_windows, verbose=0)
  return proba


def predict_multiplet(
    angle_windows_list: list[np.ndarray],
    model: keras.Model,
) -> list[np.ndarray]:
  """
  Применяет модель ко всем углам мультиплета.

  Параметры:
  angle_windows_list : list of np.ndarray (n_windows, 4000, 2)
    Список массивов окон для каждого угла.
  model : keras.Model
    Загруженная модель.

  Возврат:
  proba_list : list of np.ndarray (n_windows, 3)
    Список вероятностей для каждого угла.
  """
  proba_list = [predict_on_windows(w, model) for w in angle_windows_list]
  return proba_list


def predict_classes_from_proba(proba: np.ndarray) -> np.ndarray:
  """
  Возвращает номера классов с максимальной вероятностью.
  <<0>> --- циркуляция
  <<1>> --- либрация
  <<2>> --- переходный режим
  Параметры:
  proba : np.ndarray, shape (..., 3)
    Массив вероятностей.

  Возврат:
  classes : np.ndarray, shape (...,)
    Индексы классов 0, 1 или 2.
  """
  return np.argmax(proba, axis=-1)

#Нелбходима возможность работы со всем набором вероятностей