#Агрегировать веротятности подынтервалов
#пока в грубом виде, без работы со всем массивом вероятностей для подынтервала, учитвающиего все возможные вероятности каждого класса и задлачи порогов 
import numpy as np

def average_overlapping_probs(probs_list, window_bounds, step=None):
    """
    Усредняет вероятности для подынтервалов, образующихся при перекрытии окон.

    Подынтервал — это отрезок между началом текущего окна и началом следующего.
    Первый и последний подынтервалы покрыты только одним окном, остальные — двумя.

    Параметры:
        probs_list : list of np.ndarray (n_windows, 3)
            Предсказанные вероятности для каждого окна и каждого угла.
        window_bounds : list of tuple (start_idx, end_idx)
            Границы окон в индексах исходного временного ряда.
        step : int, optional
            Шаг между началами окон. Если не указан, вычисляется автоматически
            как window_bounds[1][0] - window_bounds[0][0].

    Возвращает:
        dict с ключами:
            'probs' : list of np.ndarray (n_subintervals, 3)
                Усреднённые вероятности для каждого угла.
            'classes' : list of np.ndarray (n_subintervals,)
                Метки классов (argmax) для каждого угла.
            'bounds' : list of tuple (start_idx, end_idx)
                Границы подынтервалов в индексах исходного ряда.
            'centers' : list of int (n_subintervals,)
                Индексы центров подынтервалов (середина отрезка).
    """
    if len(window_bounds) < 2:
        # Только одно окно — подынтервалы совпадают с окнами
        n_angles = len(probs_list)
        return {
            'probs': [p.copy() for p in probs_list],
            'classes': [np.argmax(p, axis=1) for p in probs_list],
            'bounds': window_bounds.copy(),
            'centers': [(b[0] + b[1] - 1) // 2 for b in window_bounds]
        }

    # Определяем шаг между началами окон (сейчас 1/2 окна), попытка сделать универсальный вариант для долей перекрытий
    if step is None:
        step = window_bounds[1][0] - window_bounds[0][0]

    # Количество окон
    n_windows = len(window_bounds)
    # Последний индекс исходного ряда (конец последнего окна)
    total_length = window_bounds[-1][1]

    # Строим границы подынтервалов: от начала первого окна до начала второго,
    # от начала второго до начала третьего, ..., от начала последнего окна до конца ряда.
    sub_bounds = []
    for i in range(n_windows - 1):
        start = window_bounds[i][0]
        end = window_bounds[i+1][0]
        sub_bounds.append((start, end))
    # Последний подынтервал
    sub_bounds.append((window_bounds[-1][0], total_length))

    n_sub = len(sub_bounds)  # n_windows

    n_angles = len(probs_list)
    avg_probs = []
    avg_classes = []
    
    for angle_probs in probs_list:
        # angle_probs: (n_windows, 3)
        avg = np.zeros((n_sub, 3), dtype=np.float32)
        # Первый подынтервал (только окно 0)
        avg[0] = angle_probs[0]
        # Последний подынтервал (только последнее окно)
        avg[-1] = angle_probs[-1]
        # Промежуточные подынтервалы (два окна)
        for i in range(1, n_sub - 1):
            # подынтервал i покрыт окнами i-1 и i
            avg[i] = (angle_probs[i-1] + angle_probs[i]) / 2.0 # как среднее ариф.
        avg_probs.append(avg)
        avg_classes.append(np.argmax(avg, axis=1))
    
    # Центры подынтервалов (индексы)
    centers = [(b[0] + b[1] - 1) // 2 for b in sub_bounds]
    
    return {
        'probs': avg_probs,
        'classes': avg_classes,
        'bounds': sub_bounds,
        'centers': centers
    }