# -*- coding: utf-8 -*-
import time
import numpy as np
import math
from scipy.optimize import linprog
from collections import defaultdict
import json
import pulp
from numba import njit

# --- КОНСТАНТЫ ---
EPS = 1e-8
MAX_PRICING_ITERS = 1000

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def safe_linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='highs', integrality=None,
                 options=None):
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method,
                      integrality=integrality, options=options)
    except TypeError:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method, options=options)
    return res


def extract_duals(res, n_rows):
    if hasattr(res, 'ineqlin') and getattr(res.ineqlin, 'marginals', None) is not None:
        marg = np.array(res.ineqlin.marginals)
        if len(marg) >= n_rows: return -marg[:n_rows]
    if isinstance(res, dict) and 'marginals' in res:
        marg = np.array(res['marginals'])
        if len(marg) >= n_rows: return -marg[:n_rows]
    return None

# --- Решатель задачи о рюкзаке (Numba) ---

@njit
def solve_pricing_numba(stock_length, lengths, duals, allowed_indices):
    n = len(lengths)
    dp = np.full(stock_length + 1, -1e18)
    dp[0] = 0.0
    choice = np.full(stock_length + 1, -1, dtype=np.int32)
    for cap in range(1, stock_length + 1):
        best_val = -1e18
        best_choice = -1
        for i in allowed_indices:
            li = int(lengths[i])
            if li <= cap and dp[cap - li] > -1e17:
                val = dp[cap - li] + duals[i]
                if val > best_val:
                    best_val = val
                    best_choice = i
        dp[cap] = best_val
        choice[cap] = best_choice
    max_value = -1e18
    cap_max = -1
    for cap in range(stock_length + 1):
        if dp[cap] > max_value:
            max_value = dp[cap]
            cap_max = cap
    if not np.isfinite(max_value): return None, None
    reduced_cost = 1.0 - max_value
    if reduced_cost >= -EPS: return None, None
    pattern = np.zeros(n, dtype=np.float64)
    curr_cap = cap_max
    while curr_cap > 0:
        item_idx = choice[curr_cap]
        if item_idx == -1: break
        pattern[item_idx] += 1
        curr_cap -= int(lengths[item_idx])
    return pattern, reduced_cost

def solve_pricing(stock_length, lengths, duals, allowed_indices=None):
    if allowed_indices is None:
        allowed_indices = np.arange(len(lengths), dtype=np.int32)
    else:
        allowed_indices = np.array(allowed_indices, dtype=np.int32)
    return solve_pricing_numba(stock_length, lengths, duals, allowed_indices)

# --- (остальной код остается без изменений) ---

def generate_initial_patterns(lengths, stock_length, max_combo=3):
    n = len(lengths)
    patterns = []
    from itertools import combinations
    for i in range(n):
        if lengths[i] > 0:
            cnt = stock_length // int(lengths[i])
            if cnt > 0:
                p = np.zeros(n, dtype=float)
                p[i] = cnt
                patterns.append(p)
    for r in range(2, min(max_combo, n) + 1):
        for comb in combinations(range(n), r):
            dummy_duals = np.zeros(n)
            for idx in comb:
                dummy_duals[idx] = 1.0
            res = solve_pricing(stock_length, lengths, dummy_duals, allowed_indices=comb)
            if res is not None and res[0] is not None:
                pattern, _ = res
                if np.sum(pattern) > 0:
                    patterns.append(pattern)
    if not patterns:
        return []
    A = np.array(patterns).T
    A_unique = np.unique(A, axis=1)
    return [A_unique[:, i] for i in range(A_unique.shape[1])]


def solve_integer_min_bars_with_pulp(A, demands, timeout):
    m = A.shape[1]
    n = A.shape[0]
    prob = pulp.LpProblem('min_bars', pulp.LpMinimize)
    x = [pulp.LpVariable(f'x_{j}', lowBound=0, cat='Integer') for j in range(m)]
    prob += pulp.lpSum(x)
    for i in range(n):
        prob += pulp.lpSum(A[i, j] * x[j] for j in range(m)) >= demands[i]
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=timeout))
    if pulp.LpStatus[prob.status] != 'Optimal': return False, None, None
    x_vals = np.array([pulp.value(xj) for xj in x], dtype=float)
    obj = float(sum(x_vals))
    return True, x_vals, obj


def minimize_unique_patterns_with_pulp(A, demands, K, timeout):
    m = A.shape[1]
    n = A.shape[0]
    M = int(sum(demands))
    prob = pulp.LpProblem('min_unique', pulp.LpMinimize)
    x = [pulp.LpVariable(f'x_{j}', lowBound=0, cat='Integer') for j in range(m)]
    y = [pulp.LpVariable(f'y_{j}', lowBound=0, upBound=1, cat='Binary') for j in range(m)]
    prob += pulp.lpSum(y)
    for i in range(n):
        prob += pulp.lpSum(A[i, j] * x[j] for j in range(m)) >= demands[i]
    prob += pulp.lpSum(x) == int(K)
    for j in range(m): prob += x[j] <= M * y[j]
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=timeout))
    if pulp.LpStatus[prob.status] != 'Optimal': return False, None, None
    x_vals = np.array([pulp.value(xj) for xj in x], dtype=float)
    return True, x_vals, None


def solve_csp_rich(lengths, demands, stock_length, saw_kerf, profile_name='Unknown', timeout=300):
    lengths_with_kerf = np.array(lengths, dtype=float) + saw_kerf
    demands_np = np.array(demands, dtype=float)
    n = len(lengths)
    init_patterns = generate_initial_patterns(lengths_with_kerf, stock_length, max_combo=3)
    if not init_patterns:
        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            if lengths_with_kerf[i] > 0: A[i, i] = stock_length // int(lengths_with_kerf[i])
    else:
        A = np.column_stack(init_patterns)

    if A.shape[1] == 0 or np.sum(A) == 0:
        return {'error': 'Не удалось создать начальные карты раскроя.'}

    c = np.ones(A.shape[1], dtype=float)
    res_lp = None
    for it in range(MAX_PRICING_ITERS):
        res = safe_linprog(c, A_ub=-A, b_ub=-demands_np, bounds=(0, None), method='highs',
                           options={'timeout': timeout})
        if not res.success: break
        res_lp = res
        duals = extract_duals(res, n_rows=n)
        if duals is None: break
        pricing, _ = solve_pricing(stock_length, lengths_with_kerf, duals)
        if pricing is None: break
        new_pat = pricing
        if np.any(np.all(np.isclose(A, new_pat.reshape(-1, 1)), axis=0)): break
        A = np.hstack((A, new_pat.reshape(-1, 1)))
        c = np.append(c, 1.0)

    K_lb = int(math.ceil(res_lp.fun - EPS)) if res_lp and res_lp.success else None
    
    ok_int, x_int, obj_int = solve_integer_min_bars_with_pulp(A, demands_np, timeout)
    best_K = int(math.ceil(obj_int - EPS)) if ok_int else K_lb

    used_patterns = []
    if best_K is not None:
        ok_min, x_vals, _ = minimize_unique_patterns_with_pulp(A, demands_np, best_K, timeout)
        
        if not ok_min and ok_int:
            for j in range(A.shape[1]):
                if x_int[j] > 0.5: used_patterns.append({'pattern': A[:, j].tolist(), 'count': int(round(x_int[j]))})
        elif ok_min:
            for j in range(A.shape[1]):
                if x_vals[j] > 0.5: used_patterns.append({'pattern': A[:, j].tolist(), 'count': int(round(x_vals[j]))})

    return {
        'min_bars': best_K,
        'min_unique_patterns': len(used_patterns),
        'used_patterns': used_patterns,
        'total_patterns_generated': A.shape[1]
    }


def generate_final_cutting_plan(result, original_demands, original_lengths, stock_length, saw_kerf):
    if 'used_patterns' not in result or not result['used_patterns']:
        return result

    demands_map = {i: qty for i, qty in enumerate(original_demands)}
    fulfilled_map = defaultdict(int)
    
    # ИЗМЕНЕНИЕ: Рассчитываем остаток для каждого уникального паттерна
    patterns_with_waste = []
    for p_info in result['used_patterns']:
        pattern_vector = np.array(p_info['pattern'])
        num_cuts = np.sum(pattern_vector)
        length_of_pieces = np.sum(pattern_vector * np.array(original_lengths))
        length_of_kerfs = num_cuts * saw_kerf if num_cuts > 0 else 0
        total_length_used_on_bar = length_of_pieces + length_of_kerfs
        waste_on_bar = stock_length - total_length_used_on_bar
        
        patterns_with_waste.append({
            'pattern': p_info['pattern'],
            'count': p_info['count'],
            'waste': waste_on_bar
        })

    # ИЗМЕНЕНИЕ: Сортируем по возрастанию остатка (самые эффективные - первые)
    final_individual_layouts = []
    for p_info in sorted(patterns_with_waste, key=lambda p: p['waste']):
        for _ in range(p_info['count']):
            final_individual_layouts.append(p_info['pattern'])

    corrected_layouts = []
    for layout_pattern in final_individual_layouts:
        new_layout_pattern = np.zeros_like(np.array(layout_pattern))
        is_anything_cut = False
        for i, num_pieces in enumerate(layout_pattern):
            if num_pieces == 0: continue
            needed = demands_map.get(i, 0) - fulfilled_map[i]
            if needed <= 0: continue
            can_cut = min(int(num_pieces), needed)
            new_layout_pattern[i] = can_cut
            fulfilled_map[i] += can_cut
            is_anything_cut = True

        if is_anything_cut:
            corrected_layouts.append(new_layout_pattern.tolist())

    grouped_patterns = defaultdict(int)
    for layout in corrected_layouts:
        grouped_patterns[tuple(layout)] += 1

    final_patterns = [{'pattern': list(p), 'count': c} for p, c in grouped_patterns.items()]
    new_result = result.copy()
    new_result['used_patterns'] = final_patterns
    new_result['min_bars'] = sum(p['count'] for p in final_patterns)
    new_result['min_unique_patterns'] = len(final_patterns)

    return new_result


def run_solver(input_data, stock_length, saw_kerf, timeout=300):
    output = {}
    for item in input_data:
        for material, profiles in item.items():
            output[material] = {}
            for profile, pieces in profiles.items():
                lengths = [p['length'] for p in pieces]
                demands = [p['quantity'] for p in pieces]

                result = solve_csp_rich(lengths, demands, stock_length, saw_kerf, profile_name=profile, timeout=timeout)
                # Передаем доп. параметры для правильной сортировки
                result = generate_final_cutting_plan(result, demands, lengths, stock_length, saw_kerf)

                if 'used_patterns' in result and result['used_patterns']:
                    original_lengths_np = np.array(lengths)
                    total_waste = 0.0
                    min_bars_val = result.get('min_bars')
                    if isinstance(min_bars_val, (int, float)) and min_bars_val > 0:
                        total_stock_bars_length = min_bars_val * stock_length
                        for pat_info in result['used_patterns']:
                            pattern_vector = np.array(pat_info['pattern'])
                            num_cuts = np.sum(pattern_vector)
                            length_of_pieces = np.sum(pattern_vector * original_lengths_np)
                            length_of_kerfs = num_cuts * saw_kerf if num_cuts > 0 else 0
                            total_length_used_on_bar = length_of_pieces + length_of_kerfs
                            waste_on_bar = stock_length - total_length_used_on_bar

                            pat_info['length_of_pieces'] = float(length_of_pieces)
                            pat_info['total_cuts_on_bar'] = int(num_cuts)
                            pat_info['length_of_kerfs'] = float(length_of_kerfs)
                            pat_info['total_length_used_on_bar'] = float(total_length_used_on_bar)
                            pat_info['waste_on_bar'] = float(waste_on_bar)

                            if isinstance(pat_info.get('count'), (int, float)):
                                total_waste += waste_on_bar * pat_info['count']

                        if total_stock_bars_length > 0:
                            result['total_waste'] = float(total_waste)
                            result['total_waste_percentage'] = (total_waste / total_stock_bars_length) * 100

                output[material][profile] = result
    return output

