import heapq

# Parameters
M = 16492  # Memory capacity

averaged_latency_list = []

for i in range(10):
    num_rows = int(1000 * (i + 1))
    df = dfmain.head(num_rows)
    # Initialize simulation state
    current_time = 0.0
    event_queue = []
    requests = []
    num_of_task_over_time = []

    class Request:
        def __init__(self, request_id, arrival_time, input_size, output_size):
            self.request_id = request_id
            self.arrival_time = arrival_time
            self.input_size = int(input_size)
            self.output_size = int(output_size)
            self.tokens_processed = 0
            self.memory_usage = 0
            self.started = False
            self.completed = False
            self.finish_time = None
            self.start_time = None
            self.remaining_tokens = self.output_size + 1
            self.context_length = 0

    # Populate requests and arrival events
    for idx, row in df.iterrows():
        req = Request(idx, row['arrival_time'], row['input'], row['output'])
        requests.append(req)
        event_queue.append((row['arrival_time'], 'arrival', idx))

    # Prepare for event-driven simulation
    event_queue.sort()
    waiting_prompts = set()
    running_requests = set()
    completed_requests = set()
    tokens_ready = set()
    memory_usage_over_time = []
    machine_busy = False
    batch_end_time = 0.0
    batch_in_progress = None

    heapq.heapify(event_queue)

    # Run simulation
    while event_queue or machine_busy:
        # Determine next event
        if machine_busy and (not event_queue or batch_end_time <= event_queue[0][0]):
            current_time = batch_end_time
            batch = batch_in_progress
            for req_id in batch['requests']:
                req = requests[req_id]
                if not req.started:
                    req.started = True
                    req.tokens_processed += 1
                    req.memory_usage = req.input_size + req.tokens_processed
                    req.start_time = current_time
                    tokens_ready.add(req_id)
                    running_requests.add(req_id)
                    req.remaining_tokens -= 1
                    req.context_length = 1
                else:
                    req.tokens_processed += 1
                    req.memory_usage = req.input_size + req.tokens_processed
                    req.remaining_tokens -= 1
                    if req.remaining_tokens > 0:
                        tokens_ready.add(req_id)
                        req.context_length += 1
                    else:
                        req.completed = True
                        req.finish_time = current_time
                        completed_requests.add(req_id)
                        running_requests.remove(req_id)
                        req.memory_usage = 0
            machine_busy = False
            batch_in_progress = None
        else:
            current_time, event_type, data = heapq.heappop(event_queue)
            if event_type == 'arrival':
                waiting_prompts.add(data)

        # If machine idle, form next batch
        if not machine_busy:
            batch_jobs = []
            batch_request_ids = set()

            # 1) Include ready tokens
            for req_id in list(tokens_ready):
                if req_id not in batch_request_ids:
                    req = requests[req_id]
                    batch_jobs.append({
                        'req_id': req_id,
                        'job_type': 'token',
                        'context_length': req.context_length,
                        'input_size': 0
                    })
                    batch_request_ids.add(req_id)
                    tokens_ready.remove(req_id)

            temp_running_requests = running_requests.copy()
            temp_running_requests.update(batch_request_ids)

            # 2) FCFS: sort waiting_prompts by arrival_time
            waiting_prompts_list = sorted(
                waiting_prompts,
                key=lambda x: requests[x].arrival_time
            )
            for req_id in waiting_prompts_list:
                if req_id in batch_request_ids:
                    continue

                # Tentatively add prompt
                req = requests[req_id]
                batch_jobs.append({
                    'req_id': req_id,
                    'job_type': 'prompt',
                    'context_length': 0,
                    'input_size': req.input_size
                })
                batch_request_ids.add(req_id)
                waiting_prompts.remove(req_id)
                temp_running_requests.add(req_id)

                # Memory lookahead
                ending_points = {
                    current_time + ((req_r.output_size + 1 - req_r.tokens_processed)
                                    * average_batch_processing_time)
                    for rid in temp_running_requests
                    for req_r in [requests[rid]]
                }

                memory_ok = True
                for t_prime in sorted(ending_points):
                    total_mem = 0
                    for rid in temp_running_requests:
                        req_r = requests[rid]
                        remaining = req_r.output_size + 1 - req_r.tokens_processed
                        if current_time + remaining * average_batch_processing_time >= t_prime - 1e-6:
                            tokens_done = req_r.tokens_processed + int((t_prime - current_time)
                                                                       // average_batch_processing_time)
                            tokens_done = min(tokens_done, req_r.output_size + 1)
                            total_mem += req_r.input_size + tokens_done
                    if total_mem > M:
                        memory_ok = False
                        break

                if not memory_ok:
                    temp_running_requests.remove(req_id)
                    batch_jobs.pop()
                    batch_request_ids.remove(req_id)
                    waiting_prompts.add(req_id)
                    break

            # Dispatch batch if non-empty
            if batch_jobs:
                total_context = sum(job['context_length'] for job in batch_jobs)
                batch_size = len(batch_jobs)
                avg_context = total_context / batch_size
                total_input = sum(job['input_size'] for job in batch_jobs)
                processing_time = (
                    (0.0027 * avg_context + 0.52) * batch_size +
                    44.6 + 0.378 * total_input
                ) / 1000
                batch_end_time = current_time + processing_time
                machine_busy = True
                batch_in_progress = {
                    'start_time': current_time,
                    'end_time': batch_end_time,
                    'requests': [job['req_id'] for job in batch_jobs],
                    'size': batch_size
                }

        # Track memory and tasks
        mem_sum = sum(req.input_size + req.tokens_processed
                      for req in requests if req.started and not req.completed)
        memory_usage_over_time.append((current_time, mem_sum))
        num_of_task_over_time.append(
            sum(1 for req in requests if req.started and not req.completed)
        )

    # Compute average latency
    total_latency = sum(req.finish_time - req.arrival_time
                        for req in requests if req.completed)
    count = sum(1 for req in requests if req.completed)
    averaged_latency_list.append(total_latency / count)

# Print results
print(averaged_latency_list)
