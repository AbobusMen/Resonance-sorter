import numpy as np
import pandas as pd
from pathlib import Path


def convert_angles_to_xy(angles_deg: np.ndarray) -> np.ndarray:
    """
    Переводит углы из градусов в пары (cos, sin).

    Параметры:
        angles_deg : np.ndarray, shape (...,)
            Углы в градусах.

    Возвращает:
        xy : np.ndarray, shape (..., 2), float32
    """
    angle_rad = np.deg2rad(angles_deg, dtype=np.float32)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.stack([cos_a, sin_a], axis=-1)


def _load_arrays_from_npz(filepath: str, keys: list[str]) -> dict[str, np.ndarray]:
    """
    Внутренняя функция: загружает указанные ключи из .npz-файла.
    Проверяет наличие всех ключей и возвращает их как словарь массивов float32.
    """
    data = np.load(filepath, allow_pickle=False)
    missing = set(keys) - set(data.keys())
    if missing:
        raise KeyError(f"В файле {filepath} отсутствуют ключи: {missing}")
    result = {}
    for key in keys:
        arr = data[key].astype(np.float32, copy=False)
        result[key] = arr
    return result


def load_angles(
    filepath: str,
    angle_columns: list[str],
    time_column: str | None = None,
    angle_unit: str = 'deg',
    convert_to_xy: bool = False,
) -> dict:
    """
    Загружает угловые колонки из CSV или NPZ, опционально время.

    Параметры:
        filepath : str
            Путь к файлу (.csv или .npz).
        angle_columns : list of str
            Список имён колонок с углами.
        time_column : str | None
            Имя колонки времени (None — не загружать).
        angle_unit : str
            'deg' или 'rad'. Единицы измерения во входном файле.
        convert_to_xy : bool
            Если True, добавляет (cos, sin) представление углов.

    Возвращает:
        dict с ключами:
            'angles_deg' : list of np.ndarray (N,)  — углы в градусах
            'angle_names' : list of str
            'angles_xy' : list of np.ndarray (N,2) (только при convert_to_xy=True)
            'time_years' : np.ndarray (N,) (если time_column задан)
    """
    filepath = Path(filepath)
    if filepath.suffix in ('.npz', '.npy'):
        # Загрузка из .npz (или одиночный .npy для одного угла)
        #Для единообразия пусть NPY не используется для углов)
        all_keys = list(angle_columns)
        if time_column is not None:
            all_keys.append(time_column)
        arrays = _load_arrays_from_npz(str(filepath), all_keys)
        angles_raw = [arrays[name] for name in angle_columns]

        # Приведение к градусам в зависимости от angle_unit
        if angle_unit == 'rad':
            angles_deg_list = [np.rad2deg(arr, dtype=np.float32) for arr in angles_raw]
        elif angle_unit == 'deg':
            angles_deg_list = angles_raw
        else:
            raise ValueError("angle_unit должен быть 'deg' или 'rad'")

        result = {
            'angles_deg': angles_deg_list,
            'angle_names': angle_columns,
        }
        if convert_to_xy:
            result['angles_xy'] = [convert_angles_to_xy(a) for a in angles_deg_list]
        if time_column is not None:
            result['time_years'] = arrays[time_column]
        return result

    # --- загрузка из CSV ---
    df = pd.read_csv(filepath)
    missing = set(angle_columns) - set(df.columns)
    if missing:
        raise KeyError(f"В CSV отсутствуют колонки углов: {missing}")

    angles_deg_list = []
    angles_xy_list = []

    for col in angle_columns:
        raw = df[col].values.astype(np.float32)

        if angle_unit == 'deg':
            angle_deg = raw
        elif angle_unit == 'rad':
            angle_deg = np.rad2deg(raw, dtype=np.float32)
        else:
            raise ValueError("angle_unit должен быть 'deg' или 'rad'")

        angles_deg_list.append(angle_deg)

        if convert_to_xy:
            xy = convert_angles_to_xy(angle_deg)
            angles_xy_list.append(xy)

    result = {
        'angles_deg': angles_deg_list,
        'angle_names': angle_columns,
    }
    if convert_to_xy:
        result['angles_xy'] = angles_xy_list
    if time_column is not None:
        if time_column not in df.columns:
            raise KeyError(f"Колонка времени '{time_column}' не найдена")
        result['time_years'] = df[time_column].values.astype(np.float32)

    return result


def load_period_ratio(
    filepath: str,
    period_column: str = 'period_ratio',
) -> np.ndarray:
    """
    Загружает колонку отношения периодов из CSV или NPZ.

    Возвращает:
        np.ndarray (N,) dtype float32
    """
    filepath = Path(filepath)
    if filepath.suffix in ('.npz', '.npy'):
        # Для NPZ используем _load_arrays_from_npz
        arrays = _load_arrays_from_npz(str(filepath), [period_column])
        return arrays[period_column]

    # CSV
    df = pd.read_csv(filepath)
    if period_column not in df.columns:
        raise KeyError(f"Колонка '{period_column}' не найдена в CSV")
    return df[period_column].values.astype(np.float32)


def load_multi_csv(
    filepath: str,
    angle_columns: list[str],
    period_column: str | None = 'period_ratio',
    time_column: str | None = 'time_years',
    angle_unit: str = 'deg',
    convert_to_xy: bool = False,
) -> dict:
    """
    Загружает CSV или NPZ с углами, отношением периодов и временем.
    period_column и time_column можно отключить, передав None.

    Возвращает словарь, объединяющий результаты load_angles и load_period_ratio.
    """
    angles_data = load_angles(
        filepath, angle_columns, time_column=time_column,
        angle_unit=angle_unit, convert_to_xy=convert_to_xy
    )

    result = {
        'angles_deg': angles_data['angles_deg'],
        'angle_names': angles_data['angle_names'],
    }
    if convert_to_xy:
        result['angles_xy'] = angles_data['angles_xy']
    if time_column is not None:
        result['time_years'] = angles_data['time_years']

    if period_column is not None:
        result['period_ratio'] = load_period_ratio(filepath, period_column)

    return result