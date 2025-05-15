def check_feasible(t, S, U_candidate, M, s):
    critical_points = set()
    # Critical points from ongoing requests: p + o for each in S.
    for (p, o_val) in S.values():
        critical_points.add(p + o_val)
    # Critical points from new candidate requests: t + o for each candidate.
    for (_, o_val) in U_candidate:
        critical_points.add(t + o_val)
    
    for x in sorted(critical_points):
        total_usage = 0
        # Ongoing requests: usage = s + (x - p) if x <= p + o, else 0.
        for (p, o_val) in S.values():
            if x <= p + o_val:
                usage = s + (x - p)
            else:
                usage = 0
            total_usage += usage
        # New candidate requests: usage = s + (x - t) if x <= t + o, else 0.
        for (_, o_val) in U_candidate:
            if x <= t + o_val:
                usage = s + (x - t)
            else:
                usage = 0
            total_usage += usage
        if total_usage > M:
            return False
    return True

# -------------------------------
# Online Semi-Online Scheduling
# -------------------------------
def online_semi_online_scheduling(M, arrivals, s):
    """
    Schedules jobs in an online manner.
    
    arrivals: a list of dicts, each with keys:
       - 'arrival_time': when the job arrives.
       - 'length': the output length (o) of the job.
    
    s: fixed prompt size.
    
    Returns:
      start_times: a dictionary mapping job id to its scheduled start time.
      total_latency: the sum over jobs of (start time + length - arrival_time).
    """
    # Number of jobs.
    n = len(arrivals)
    
    # Create a list of (job_id, arrival dict) and sort by arrival time.
    sorted_arrivals = sorted(list(enumerate(arrivals)), key=lambda x: x[1]['arrival_time'])
    arrival_index = 0  # pointer into sorted_arrivals
    
    R = []  # List of available (unscheduled) jobs, each as a tuple (job_id, length)
    S = {}  # Ongoing scheduled jobs: job_id -> (start time, length)
    start_times = {}  # To record scheduled start times.
    t = 0  # Discrete time counter.
    
    # Continue until all jobs have arrived and been scheduled, and no job is running.
    while (arrival_index < n) or R or S:
        # Add any new arrivals that have arrived by time t.
        while arrival_index < n and sorted_arrivals[arrival_index][1]['arrival_time'] <= t:
            job_id, job_info = sorted_arrivals[arrival_index]
            R.append((job_id, job_info['length']))
            arrival_index += 1
        
        # Remove finished jobs from ongoing set S.
        finished = [job_id for job_id, (p, o_val) in S.items() if t >= p + o_val]
        for job_id in finished:
            del S[job_id]
        
        # Try to add as many available jobs (from R) as possible.
        # Sort available jobs by output length (and then job id to break ties).
        R.sort(key=lambda x: (x[1], x[0]))
        U = []  # Candidate batch to schedule at time t.
        for candidate in R:
            U_candidate = U + [candidate]
            if check_feasible(t, S, U_candidate, M, s):
                U = U_candidate
            else:
                # Cannot add further jobs without violating memory constraint.
                break
        
        # Schedule all jobs in the candidate batch U with start time = t.
        for (job_id, o_val) in U:
            start_times[job_id] = t
            S[job_id] = (t, o_val)
        # Remove scheduled jobs from R.
        scheduled_ids = {job_id for (job_id, _) in U}
        R = [job for job in R if job[0] not in scheduled_ids]
        
        # Advance time.
        t += 1
    
    # Compute total latency.
    total_latency = 0
    for job_id, job in enumerate(arrivals):
        # For each job: latency = (start time + length) - arrival_time.
        finish_time = start_times[job_id] + job['length']
        latency = finish_time - job['arrival_time']
        total_latency += latency
    
    return start_times, total_latency
