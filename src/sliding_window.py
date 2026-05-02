import numpy as np

def sliding_windows_multiplet(
  data_dict: dict,
  window_length: int = 4000,
  overlap_ratio: float = 0.0,
) -> dict:
  """
  Нарезает временные ряды углов на окна.

  Параметры
  data_dict : dict
    Словарь, возвращаемый load_multiplet_csv. Ожидаемые ключи:
    - 'angles_xy' : list of np.ndarray (N, 2) для каждого угла
    - 'period_ratio' : np.ndarray (N,) (сохраняется, но не агрегируется)
  window_length : int
    Длина одного окна (по умолчанию 4000).
  overlap_ratio : float
    Доля перекрытия между соседними окнами в диапазоне [0.0, 1.0).
    0.0 — без перекрытия, 0.5 — перекрытие на половину.

  Returns
  -------
  windows_data : dict
    Словарь с ключами:
    - 'angle_windows' : list of np.ndarray (n_windows, window_length, 2)
                        Список окон для каждого угла.
    - 'window_bounds' : list of tuples (start_idx, end_idx)
                        Индексы границ каждого окна в исходном массиве.
  """
  if not (0.0 <= overlap_ratio < 1.0):
    raise ValueError("overlap_ratio должен быть в диапазоне [0.0, 1.0)")

  if 'angles_xy' not in data_dict:
    raise KeyError("data_dict должен содержать ключ 'angles_xy'")

  angles_xy_list = data_dict['angles_xy']

  if len(angles_xy_list) == 0:
    raise ValueError("Список углов пуст")

  total_length = angles_xy_list[0].shape[0]
  if total_length < window_length:
    raise ValueError(f"Длина ряда ({total_length}) меньше длины окна ({window_length})")

  step = int(window_length * (1 - overlap_ratio))
  if step == 0:
    raise ValueError("overlap_ratio слишком велик – шаг окон стал нулевым")

  n_windows = (total_length - window_length) // step + 1

  angle_windows_list = []
  window_bounds = []

  for angle_arr in angles_xy_list:
    if angle_arr.shape[0] != total_length:
      raise ValueError("Все угловые массивы должны иметь одинаковую длину")
    windows = np.zeros((n_windows, window_length, 2), dtype=np.float32)
    for i in range(n_windows):
      start = i * step
      end = start + window_length
      windows[i] = angle_arr[start:end]
    angle_windows_list.append(windows)

  # Сохраняем границы окон (одинаковы для всех углов)
  for i in range(n_windows):
    start = i * step
    end = start + window_length
    window_bounds.append((start, end))

  return {
    'angle_windows': angle_windows_list,
    'window_bounds': window_bounds
  }