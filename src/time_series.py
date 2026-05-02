import numpy as np
import pandas as pd


def load_multi_csv(
  filepath: str,
  angle_columns: list[str],
  period_column: str = 'period_ratio',
  time_column: str = 'time_years', 
  angle_unit: str = 'deg',
  convert_to_xy: bool = True,
) -> dict[str, np.ndarray | list[np.ndarray] | list[str]]:
  """
  Загрузка CSV с period_ratio, phi_0_original, phi_1_original, ...
  P_out/P_in и мультиплет двутельного MMR.

  Параметры:
  filepath : str
    Путь к CSV-файлу.
  angle_columns : list of str
    Список имён колонок с углами.
  period_column : str
    Колонка с отношением периодов (внешний/внутренний) (по умолчанию 'period_ratio').
  angle_unit : str
    'deg' — углы в градусах, 'rad' — в радианах.
  convert_to_xy : bool
    Если True, возвращает (cos a, sin a) для каждого угла a.

  Возвращает:
  dict
    Словарь с ключами:
    - 'period_ratio': np.ndarray (N,)
    - 'angles_deg': list of np.ndarray (N,) — углы в градусах
    - 'angles_xy': list of np.ndarray (N, 2) — только если convert_to_xy=True
    - 'angle_names': list of str
  """
  df = pd.read_csv(filepath)

  required = [period_column] + angle_columns
  missing = set(required) - set(df.columns)
  if missing:
    raise KeyError(f"В CSV отсутствуют колонки: {missing}")

  period_ratio = df[period_column].values.astype(np.float32)
  time_years = df[time_column].values.astype(np.float32)

  angles_deg_list = []
  angles_xy_list = []

  for col in angle_columns:
    raw = df[col].values.astype(np.float32)#np.float64

    if angle_unit == 'deg':
      angle_deg = raw
    elif angle_unit == 'rad':
      angle_deg = np.rad2deg(raw)
    else:
      raise ValueError("angle_unit должен быть 'deg' или 'rad'")

    angles_deg_list.append(angle_deg)

    if convert_to_xy:
      angle_rad = np.deg2rad(angle_deg)
      cos_a = np.cos(angle_rad)
      sin_a = np.sin(angle_rad)
      xy = np.stack([cos_a, sin_a], axis=-1)
      angles_xy_list.append(xy)

  result = {
    'time_years': time_years, 
    'period_ratio': period_ratio,
    'angles_deg': angles_deg_list,
    'angle_names': angle_columns,
  }
  if convert_to_xy:
    result['angles_xy'] = angles_xy_list

  return result





