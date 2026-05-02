# criteria.py
import numpy as np
from collections import deque


def aggregate_labels(class_labels_list):
    """
    Агрегирует метки нескольких углов в одну последовательность
    с приоритетом: 1 (либрация) > 2 (переходный) > 0 (циркуляция).

    Параметры:
      class_labels_list : list of np.ndarray (n_windows,)
          Список массивов меток (0,1,2) для каждого угла.

    Возвращает:
      aggregated : np.ndarray (n_windows,)
    """
    stacked = np.stack(class_labels_list, axis=0)  # (n_angles, n_windows)
    aggregated = np.where(np.any(stacked == 1, axis=0), 1,
                 np.where(np.any(stacked == 2, axis=0), 2, 0))
    return aggregated


def find_resonance_segments(
  aggregated_labels,
  window_bounds,
  time_years,
  period_ratio,
  expected_ratio, # Точное (p+q)/p (Резонансное отношение периодов)
  min_librations=30,
  max_consecutive_nonlib=2,
  sliding_window_size=6,
  max_nonlib_in_sliding=2,
):
  """
  Ищет сегменты долгой либрации в агрегированной последовательности меток.

  Критерии сегмента:
    * Количество либраций в сегменте >= min_librations.
    * Внутри сегмента не может быть больше max_consecutive_nonlib не-либраций подряд.
    * В любом скользящем окне длиной sliding_window_size число не-либраций
      не должно превышать max_nonlib_in_sliding.

  Для каждого найденного сегмента вычисляется
      Δ_max = max(|period_ratio - expected_ratio|)
  по всем временным точкам, попавшим внутрь сегмента.

  Параметры:
      aggregated_labels : np.ndarray (n_windows,)
      window_bounds : list of (start_idx, end_idx)
      time_years : np.ndarray (N,)
      period_ratio : np.ndarray (N,)
      expected_ratio : float
      min_librations : int
      max_consecutive_nonlib : int
      sliding_window_size : int
      max_nonlib_in_sliding : int

  Возвращает:
      list of dict с ключами:
          'librations_count', 'start_window', 'end_window',
          't_start_years', 't_end_years', 'duration_years', 'Delta_max'
  """
  n_windows = len(aggregated_labels)
  if n_windows == 0:
      return []

  segments = []
  in_segment = False
  segment_start = 0
  lib_count = 0
  consecutive_nonlib = 0

  # Кольцевой буфер для скользящего окна
  ring_buffer = deque(maxlen=sliding_window_size)
  nonlib_in_buffer = 0

  for i in range(n_windows):
      label = aggregated_labels[i]

      if not in_segment:
          # Ждём первую либрацию, чтобы начать сегмент
          if label == 1:
              in_segment = True
              segment_start = i
              lib_count = 1
              consecutive_nonlib = 0

              # Очищаем буфер и начинаем заполнять с текущей позиции
              ring_buffer.clear()
              nonlib_in_buffer = 0
              ring_buffer.append(label)   # label == 1
      else:
          # ------------------- обновление скользящего буфера ----------
          if len(ring_buffer) == sliding_window_size:
              oldest = ring_buffer[0]
              if oldest != 1:
                  nonlib_in_buffer -= 1
              ring_buffer.append(label)
          else:
              ring_buffer.append(label)

          if label != 1:
              nonlib_in_buffer += 1

          # ------------------- обновление счётчиков -----------------
          if label == 1:
              lib_count += 1
              consecutive_nonlib = 0
          else:
              consecutive_nonlib += 1

          # ---------- ПРОВЕРКИ УСЛОВИЙ СБРОСА ----------
          # Условие 1: слишком много не-либраций подряд
          if consecutive_nonlib > max_consecutive_nonlib:
              # сегмент обрывается
              if lib_count >= min_librations:
                  segments.append(_make_segment(segment_start, i - consecutive_nonlib,
                                                lib_count, window_bounds,
                                                time_years, period_ratio, expected_ratio))
              in_segment = False
              continue

          # Условие 2: превышение не-либраций в скользящем окне
          if len(ring_buffer) == sliding_window_size:
              if nonlib_in_buffer > max_nonlib_in_sliding:
                  # сегмент обрывается на предыдущем элементе
                  if lib_count >= min_librations:
                      # lib_count мог увеличиться на текущей метке,
                      # поэтому если label != 1, то считаем lib_count без учёта текущей
                      actual_lib = lib_count if label == 1 else lib_count - 1
                      segments.append(_make_segment(segment_start, i - 1, actual_lib,
                                                    window_bounds, time_years, period_ratio,
                                                    expected_ratio))
                  in_segment = False
                  continue

  # Конец массива – закрываем сегмент, если он ещё активен
  if in_segment and lib_count >= min_librations:
      segments.append(_make_segment(segment_start, n_windows - 1, lib_count,
                                    window_bounds, time_years, period_ratio,
                                    expected_ratio))

  return segments


def _make_segment(start_win, end_win, lib_count, window_bounds, time_years, period_ratio, expected_ratio):
    t_start = time_years[window_bounds[start_win][0]]
    t_end = time_years[window_bounds[end_win][1] - 1]  # последняя точка последнего окна
    mask = (time_years >= t_start) & (time_years <= t_end)
    ratios = period_ratio[mask]

    if len(ratios) > 0:
        Delta_max = np.max(np.abs(ratios - expected_ratio))
    else:
        Delta_max = np.nan

    duration = t_end - t_start

    return {
        'librations_count': lib_count,
        'start_window': start_win,
        'end_window': end_win,
        't_start_years': t_start,
        't_end_years': t_end,
        'duration_years': duration,
        'Delta_max': Delta_max,
    }


def process_multiplet(
    class_labels_list,
    window_bounds,
    time_years,
    period_ratio,
    expected_ratio,
    min_librations=30,
    max_consecutive_nonlib=2,
    sliding_window_size=6,
    max_nonlib_in_sliding=2,
):
    """
    Агрегирует метки мультиплета и ищет резонансные сегменты.
    Параметры те же, что у aggregate_labels + find_resonance_segments.
    Возвращает список сегментов (list of dict).
    """
    agg = aggregate_labels(class_labels_list)
    return find_resonance_segments(agg, window_bounds, time_years, period_ratio,
                                   expected_ratio, min_librations,
                                   max_consecutive_nonlib, sliding_window_size,
                                   max_nonlib_in_sliding)