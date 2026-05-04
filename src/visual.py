import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


def plot_angles(
    time_years: np.ndarray,
    angles_deg_list: list[np.ndarray],
    title: str | None = None,
    save_path: str | None = None,
    fig_width: float = 10.0,
    row_height: float = 2.5,
):
    """
    Рисует временные ряды углов без отношения периодов.

    Параметры:
        time_years : np.ndarray, shape (N,)
            Время в годах.
        angles_deg_list : list of np.ndarray (N,) 
            Список углов в градусах.
        title : str, optional
            Заголовок фигуры.
        save_path : str, optional
            Путь для сохранения (например, 'angles.png').
        fig_width : float
            Ширина рисунка в дюймах.
        row_height : float
            Высота одного подграфика в дюймах.
    """
    num_angles = len(angles_deg_list)
    fig, axes = plt.subplots(
        num_angles, 1,
        figsize=(fig_width, row_height * num_angles),
        sharex=True,
    )
    if num_angles == 1:
        axes = [axes]
    #Based on a script developed by Dr. Christophe Cossou
    # Форматтер для оси X
    xfmt = ScalarFormatter(useOffset=True)
    xfmt.set_scientific(True)
    xfmt.set_powerlimits((-3, 3))

    for i, angle_deg in enumerate(angles_deg_list):
        ax = axes[i]
        ax.set_yticks(np.arange(-90, 271, 90))
        ax.set_ylim(-90, 270)
        ax.plot(time_years, angle_deg, '.', markersize=0.5, alpha=0.8)
        ax.set_ylabel(rf"$\varphi_{{{i}}}^\circ$")
        ax.grid(True)
        ax.xaxis.set_major_formatter(xfmt)

    axes[-1].set_xlabel("Время (годы)")

    if title:
        fig.suptitle(title)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    plt.show()


def plot_raw_multiplet(
    time_years: np.ndarray,
    period_ratio: np.ndarray,
    angles_deg_list: list[np.ndarray],
    title: str | None = None,
    save_path: str | None = None,
    fig_width: float = 10.0,
    row_height: float = 2.5,
):
    """
    Рисует исходные временные ряды: отношение периодов и углы мультиплета на интервале.

    Параметры:
        time_years : np.ndarray, shape (N,)
            Время в годах.
        period_ratio : np.ndarray, shape (N,)
            Отношение периодов P_out / P_in.
        angles_deg_list : list of np.ndarray, shape (N,) each
            Список углов в градусах.
        title : str, optional
            Заголовок всей фигуры.
        save_path : str, optional
            Если указан, фигура сохраняется в файл (например, 'output.png').
        fig_width : float
            Ширина рисунка в дюймах.
        row_height : float
            Высота одного подграфика в дюймах.
    """
    num_angles = len(angles_deg_list)
    nb_rows = 1 + num_angles          # первая строка под период, остальные под углы
    fig_height = row_height * nb_rows
    fig, axes = plt.subplots(nb_rows, 1, figsize=(fig_width, fig_height), sharex=True)

    # Чтобы работало и с одним подграфиком
    if nb_rows == 1:
        axes = [axes]

    # Настройка форматтера для оси X (научная нотация)
    xfmt = ScalarFormatter(useOffset=True)
    xfmt.set_scientific(True)
    xfmt.set_powerlimits((-3, 3))

    # 1) Отношение периодов
    ax_period = axes[0]
    ax_period.plot(time_years, period_ratio, linewidth=0.8)
    ax_period.set_ylabel(r"$P_{out}/P_{in}$")
    ax_period.grid(True)
    ax_period.xaxis.set_major_formatter(xfmt)

    # 2) Углы
    for i, angle_deg in enumerate(angles_deg_list):
        ax = axes[i + 1]
        ax.set_yticks(np.arange(-90, 271, 90))
        ax.set_ylim(-90, 270)
        ax.plot(time_years, angle_deg, '.', markersize=0.5, alpha=0.8)
        ax.set_ylabel(r"$\varphi_{%d}^{\circ}$" % i)
        ax.grid(True)

    # Общая подпись оси X
    axes[-1].set_xlabel("Время (годы)")

    if title:
        fig.suptitle(title)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    plt.show()


def plot_multiplet_predictions(
    time_years: np.ndarray,
    angles_deg_list: list[np.ndarray],
    window_bounds: list[tuple[int, int]],
    class_labels_list: list[np.ndarray],
    title: str | None = None,
    save_path: str | None = None,
    fig_width: float = 12.0,
    row_height: float = 2.5,
    alpha: float = 0.3,
):
    """
    Рисует углы мультиплета с заливкой окон по предсказанным классам.
    
    Параметры:
        time_years : np.ndarray, shape (N,)
            Время в годах для каждой точки исходного ряда.
        angles_deg_list : list of np.ndarray, shape (N,) each
            Список углов в градусах.
        window_bounds : list of tuple (start_idx, end_idx)
            Границы окон (индексы) из sliding_windows_multiplet.
        class_labels_list : list of np.ndarray, shape (n_windows,) each
            Предсказанные классы (0,1,2) для каждого окна и каждого угла.
        title : str, optional
            Заголовок рисунка.
        save_path : str, optional
            Путь для сохранения (если нужен).
        fig_width : float
            Ширина рисунка.
        row_height : float
            Высота одного подграфика.
        alpha : float
            Прозрачность заливки.
    """
    # Цвета классов: 0 – циркуляция (синий), 1 – либрация (зелёный), 2 – переходный (фиолетовый)
    colors = {0: 'blue', 1: 'green', 2: 'purple'}
    labels = {0: 'Циркуляция', 1: 'Либрация', 2: 'Переходный'}
    
    num_angles = len(angles_deg_list)
    fig, axes = plt.subplots(num_angles, 1, figsize=(fig_width, row_height * num_angles), sharex=True)
    if num_angles == 1:
        axes = [axes]
    
    # Для каждого угла
    #Based on a script developed by Dr. Christophe Cossou
    for i, (angle_deg, class_labels) in enumerate(zip(angles_deg_list, class_labels_list)):
        ax = axes[i]
        ax.set_yticks(np.arange(-90, 271, 90))
        ax.set_ylim(-90, 270)
        
        # Заливка окон цветом класса
        for j, (start, end) in enumerate(window_bounds):
            # Время начала и конца окна
            t_start = time_years[start]
            # Конец окна: последняя точка окна + половина интервала до следующей точки,
            # чтобы заливка была без разрывов. Если end < len(time_years), иначе продлеваем по последнему шагу.
            if end < len(time_years):
                t_end = time_years[end-1] + 0.5 * (time_years[end] - time_years[end-1])
            else:
                dt = time_years[-1] - time_years[-2]
                t_end = time_years[-1] + 0.5 * dt
            cls = class_labels[j]
            ax.axvspan(t_start, t_end, color=colors[cls], alpha=alpha, label=None)
        
        # Основной график угла (чёрные точки)
        ax.plot(time_years, angle_deg, '.', color='black', markersize=0.5, alpha=0.8)
        # ax.set_ylabel(rf"$\varphi_{{{i}}}\ (^\circ)$")
        ax.set_ylabel(rf"$\varphi_{{{i}}}^\circ$")
        ax.grid(True)
    
    # Создаём легенду вручную (один раз для всей фигуры)
    handles = [plt.Rectangle((0,0),1,1, color=colors[c], alpha=alpha) for c in [0,1,2]]
    fig.legend(handles, [labels[c] for c in [0,1,2]], loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    axes[-1].set_xlabel("Время (годы)")
    
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # оставляем место для заголовка и легенды
    
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()