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


#Непрерывная вероятность для подынтервала с перекрытием в 1/2 окна. Пока только такая реализация, без перекрытий в 1/3 и т.д.
def continuous_probabilities(probs_list, window_bounds):
    """
    Непрерывные вероятности классов для каждой точки исходного ряда
    с треугольным весовым усреднением в зонах перекрытия окон.
    

    Параметры:
        probs_list : list of np.ndarray (n_windows, 3)
            Предсказанные вероятности для каждого окна и каждого угла.
        window_bounds : list of tuple (start_idx, end_idx)
            Границы окон в индексах исходного временного ряда.
    Возвращает:
        dict с ключами:
            'probs' : list of np.ndarray (T, 3)  – вероятности для каждой точки
            'classes' : list of np.ndarray (T,)  – метки классов (argmax)
            где T – общая длина исходного ряда (конец последнего окна).
            
    Первый и последний подынтервалы заполняются вероятностями соответствующих окон без вычисления весов, т.к имеют лишь один набор вероятностей от одного окна.

            
    """
    total_length = window_bounds[-1][1]
    T = total_length
    n_angles = len(probs_list)
    n_windows = len(window_bounds)

    # Определим границы первого и последнего подынтервалов
    first_segment_end = window_bounds[1][0] if n_windows > 1 else window_bounds[0][1] # до начала второго окна
    last_segment_start = window_bounds[-1][0]   # начало последнего окна

    all_probs = []
    all_classes = []

    for angle_probs in probs_list:
        result_prob = np.zeros((T, 3), dtype=np.float32)

        # 1) Первый подынтервал: вероятности первого окна (для всех точек - одно значение)
        result_prob[:first_segment_end] = angle_probs[0]
        # 2) Последний подынтервал: вероятности последнего окна (для всех точек - одно значение)
        result_prob[last_segment_start:] = angle_probs[-1]

        # 3) Промежуточные точки
        if n_windows > 2:
            window_length = window_bounds[0][1] - window_bounds[0][0] # длина подынтервала
            half_window = window_length / 2.0 # середина подынтервала

            sum_weighted = np.zeros((T, 3), dtype=np.float32) # веса
            sum_weights = np.zeros(T, dtype=np.float32) # cумма взвешенных вероятностей 

            # Обрабатываем все окна, вклад будет добавлен и для промежуточных точек
            for idx in range(n_windows):
                start, end = window_bounds[idx]
                center = (start + end - 1) / 2.0 # середина окна-подынтервала через интексы
                t_indices = np.arange(start, end) # вектор индексов
                dist = np.abs(t_indices - center) # дистанция от центра 
                # 4000->2000->[1000] (500;500)
                weight = np.maximum(0, 1.0 - dist / half_window) # max[0,1-d/half_window]

                # TRIMA треугольное весовое усреднение
                prob = angle_probs[idx]
                sum_weighted[t_indices] += weight[:, np.newaxis] * prob[np.newaxis, :]
                sum_weights[t_indices] += weight

            # Заменяем значения в промежуточной области взвешенным средним
            middle_region = slice(first_segment_end, last_segment_start)
            valid = sum_weights[middle_region] > 0
            result_prob[middle_region][valid] = (sum_weighted[middle_region][valid] / 
                                                 sum_weights[middle_region, np.newaxis][valid])

        all_probs.append(result_prob)
        all_classes.append(np.argmax(result_prob, axis=1))

    return {'probs': all_probs, 'classes': all_classes}

    