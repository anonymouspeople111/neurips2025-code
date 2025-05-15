import random
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve_MILP_online(M, arrivals, s, processing_time=None, start_times_dict=None):
    """
    Solves the MILP for the online scheduling problem.
    
    Each job i is characterized by:
       - a_i: arrival time (from arrivals)
       - o_i: processing length (from arrivals)
    
    We define the end-to-end latency for job i as:
         latency_i = (start_time_i - a_i) + o_i.
    The goal is to minimize the total latency: ∑_i [ (start_time_i - a_i) + o_i ].
    
    The model:
      Decision variables: For job i=0,...,n-1 and time t=a_i,...,T, 
         x[i,t] ∈ {0,1} indicates if job i starts at time t.
      Let T = max_i a_i + ∑_i o_i.
      
      Objective:
         Minimize ∑_{i=0}^{n-1} [ (∑_{t=a_i}^{T} t * x[i,t] - a_i) + o_i ].
         
      Constraints:
         (1) ∀ i:  ∑_{t=a_i}^{T} x[i,t] = 1.
         (2) ∀ τ=0,...,T:  
             ∑_{i: a_i ≤ τ} ∑_{t = max(a[i], τ-o[i])}^{τ-1} (s + (τ - t)) * x[i,t] ≤ M.
    
    Parameters:
      - M: Total available memory.
      - arrivals: list of dictionaries; each dictionary must contain:
                  'arrival_time' and 'length'
      - s: Fixed prompt size.
      - processing_time: (Optional) Time limit for the solver (in seconds).
      - start_times_dict: (Optional) Warm-start solution, a dictionary mapping job i to a start time.
    
    Returns:
      A tuple (total_latency, sol_start_times), where:
         total_latency = ∑_i ((start_time_i - a_i) + o_i)
         sol_start_times is a dictionary mapping job i to its chosen start time.
      If no solution is found, returns (None, None).
    """
    n = len(arrivals)
    # Extract arrival times and processing lengths.
    a = [req['arrival_time'] for req in arrivals]
    o = [req['length'] for req in arrivals]
    
    # Time horizon: T = max(a) + sum(o)
    T = max(a) + sum(o)
    
    # Create the model.
    model = gp.Model("Online_Schedule")
    
    if processing_time is not None:
        model.setParam('TimeLimit', processing_time)
    
    # Create decision variables: x[i,t] for each job i and for t = a[i] to T.
    x = {}
    for i in range(n):
        for t in range(a[i], T+1):
            x[(i,t)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")
    model.update()
    
    # Objective: For each job i, the contribution is:
    #   (start_time_i - a[i]) + o[i], where start_time_i = sum_{t=a[i]}^T t * x[i,t].
    # Hence, the overall objective is:
    #   ∑_{i=0}^{n-1} [ (∑_{t=a[i]}^T t*x[i,t] - a[i]) + o[i] ].
    obj_expr = gp.LinExpr()
    for i in range(n):
        for t in range(a[i], T+1):
            obj_expr.addTerms(t, x[(i,t)])
        obj_expr.addConstant(-a[i] + o[i])
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Constraint (1): Each job i must start exactly once.
    for i in range(n):
        model.addConstr(gp.quicksum(x[(i,t)] for t in range(a[i], T+1)) == 1,
                        name=f"StartOnce_{i}")
    
    # Constraint (2): Memory usage constraint.
    # For each time τ from 0 to T, consider all jobs i that have arrived (a[i] ≤ τ).
    # If job i started at time t, then it is active at τ if t ≤ τ < t+o[i].
    # In our corrected formulation, we sum for t from lower_bound = max(a[i], τ-o[i])
    # up to τ-1. (Thus a job starting exactly at τ does not count.)
    for tau in range(0, T+1):
        mem_expr = gp.LinExpr()
        for i in range(n):
            if a[i] <= tau:
                lower_bound = max(a[i], tau - o[i])
                for t in range(lower_bound, tau):
                    mem_expr.addTerms(s + (tau - t), x[(i,t)])
        model.addConstr(mem_expr <= M, name=f"Mem_{tau}")
    
    # (Optional) Warm-start: apply the provided start_times_dict.
    if start_times_dict is not None:
        for i in range(n):
            init_start = start_times_dict.get(i, a[i])
            for t in range(a[i], T+1):
                x[(i,t)].start = 1 if t == init_start else 0
    model.update()
    
    # Optimize the model.
    model.optimize()
    
    if model.Status in [GRB.OPTIMAL, GRB.INTERRUPTED, GRB.TIME_LIMIT]:
        sol_start_times = {}
        for i in range(n):
            for t in range(a[i], T+1):
                if x[(i,t)].X > 0.5:
                    sol_start_times[i] = t
                    break
            if i not in sol_start_times:
                sol_start_times[i] = a[i]
        total_latency = sum(sol_start_times[i] - a[i] + o[i] for i in range(n))
        return total_latency, sol_start_times
    else:
        return None, None
