# Parameters
M = 16492  # Memory capacity

averaged_latency_list = []

for i in range(1):
    num_rows = int(1000 * (i + 1))
    df = dfmain.head(num_rows)
    # Initialize variables
    current_time = 0.0
    event_queue = []  # List of events: (time, event_type, data)

    # Create a list to hold the requests
    requests = []

    num_of_task_over_time = []

    # Create a class to represent each request
    class Request:
        def __init__(self, request_id, arrival_time, input_size, output_size):
            self.request_id = request_id
            self.arrival_time = arrival_time
            self.input_size = int(input_size)
            self.output_size = int(output_size)
            self.tokens_processed = 0  # Number of output tokens processed
            self.memory_usage = 0  # Current memory usage
            self.started = False  # Whether the prompt has been started processing
            self.completed = False  # Whether the request has been fully processed
            self.finish_time = None  # Time when the last token was processed
            self.start_time = None  # Time when the prompt was first processed
            self.remaining_tokens = self.output_size + 1  # Include prompt processing
            self.context_length = 0  # Context length for tokens

    # Create requests from df
    for idx, row in df.iterrows():
        req = Request(
            request_id=idx,
            arrival_time=row['arrival_time'],
            input_size=row['input'],
            output_size=row['output']
        )
        requests.append(req)
        # Add arrival event to the event queue
        event_queue.append((row['arrival_time'], 'arrival', idx))

    # Sort the event queue by time
    event_queue.sort()

    # Initialize system state
    waiting_prompts = set()  # Requests that have arrived but not yet started
    running_requests = set()  # Requests that have been started but not yet completed
    completed_requests = set()  # Requests that have been completed
    tokens_ready = set()  # Requests whose next token is ready to be processed

    # Initialize latency tracking
    total_latency = 0.0

    # Initialize memory usage tracking
    memory_usage_over_time = []

    # Initialize machine state
    machine_busy = False
    batch_end_time = 0.0
    batch_in_progress = None

    heapq.heapify(event_queue)  # Convert event_queue to a heap

    # Main simulation loop
    while event_queue or machine_busy:
        # Determine the next event
        if machine_busy and (not event_queue or batch_end_time <= event_queue[0][0]):
            # The next event is the batch completion
            current_time = batch_end_time
            # Process batch completion
            batch = batch_in_progress
            # Update requests
            for req_id in batch['requests']:
                req = requests[req_id]
                if not req.started:
                    # This was a prompt
                    req.started = True
                    req.tokens_processed += 1
                    req.memory_usage = req.input_size + req.tokens_processed
                    req.start_time = current_time
                    tokens_ready.add(req_id)
                    running_requests.add(req_id)
                    req.remaining_tokens -= 1
                    req.context_length = 1  # Next token will have context length 1
                else:
                    # This was a token
                    req.tokens_processed += 1
                    req.memory_usage = req.input_size + req.tokens_processed
                    req.remaining_tokens -= 1
                    if req.remaining_tokens > 0:
                        tokens_ready.add(req_id)
                        req.context_length += 1  # Increment context length
                    else:
                        # Request is completed
                        req.completed = True
                        req.finish_time = current_time
                        completed_requests.add(req_id)
                        running_requests.remove(req_id)
                        # Free memory
                        req.memory_usage = 0
                # Update context_length for the next token
            # Clear the batch
            machine_busy = False
            batch_in_progress = None
        else:
            # The next event is an arrival
            current_time, event_type, data = heapq.heappop(event_queue)
            if event_type == 'arrival':
                req_id = data
                waiting_prompts.add(req_id)
            else:
                pass

        # Check if machine is idle and can start a new batch
        if not machine_busy:
            # Form a batch according to the algorithm
            batch_jobs = []
            batch_request_ids = set()

            # Step 1: Include tokens ready to be processed
            for req_id in list(tokens_ready):
                if req_id not in batch_request_ids:
                    req = requests[req_id]
                    # It's a token
                    batch_jobs.append({
                        'req_id': req_id,
                        'job_type': 'token',
                        'context_length': req.context_length,  # Use the context length
                        'input_size': 0  # Tokens do not contribute to total input size
                    })
                    batch_request_ids.add(req_id)
                    tokens_ready.remove(req_id)

            # Initialize temp_running_requests as a copy of running_requests
            temp_running_requests = running_requests.copy()
            # Add the requests whose tokens are ready to be processed (they will process one more token)
            temp_running_requests.update(batch_request_ids)

            # Step 2: Consider waiting prompts, sorted by output size
            waiting_prompts_list = sorted(
                waiting_prompts, key=lambda x: requests[x].output_size)
            for req_id in waiting_prompts_list:
                if req_id in batch_request_ids:
                    continue

                # Tentatively add the prompt to the batch
                req = requests[req_id]
                batch_jobs.append({
                    'req_id': req_id,
                    'job_type': 'prompt',
                    'context_length': 0,  # Prompts have context length 0
                    'input_size': req.input_size  # Prompts contribute to total input size
                })
                batch_request_ids.add(req_id)
                waiting_prompts.remove(req_id)
                # Update temp_running_requests for the next iteration
                temp_running_requests.add(req_id)

                # Perform memory constraint check at completion times
                ending_points = set()
                # Collect completion times of temp_running_requests
                for rid in temp_running_requests:
                    req_r = requests[rid]
                    remaining_tokens = (req_r.output_size + 1) - req_r.tokens_processed
                    completion_time = current_time + remaining_tokens * average_batch_processing_time
                    ending_points.add(completion_time)

                # All ending points (times when any request finishes)
                ending_points = sorted(ending_points)

                memory_ok = True
                for t_prime in ending_points:
                    # Calculate memory usage at time t_prime
                    active_requests = []
                    # Existing running requests
                    for rid in temp_running_requests:
                        req_r = requests[rid]
                        remaining_tokens = (req_r.output_size + 1) - req_r.tokens_processed
                        req_completion_time = current_time + remaining_tokens * average_batch_processing_time
                        if req_completion_time >= t_prime - 1e-6:
                            active_requests.append(req_r)
                    # Calculate total memory usage
                    total_memory_usage = 0
                    for req_r in active_requests:
                        tokens_processed = req_r.tokens_processed + int((t_prime - current_time) // average_batch_processing_time)
                        tokens_processed = min(tokens_processed, req_r.output_size + 1)
                        memory_usage_req = req_r.input_size + tokens_processed
                        total_memory_usage += memory_usage_req

                    if total_memory_usage > M:
                        memory_ok = False
                        break  # Memory limit exceeded at this time

                if not memory_ok:
                    # Cannot add more prompts without exceeding memory limit
                    # Remove the prompt from temp_running_requests (since it was tentatively added)
                    temp_running_requests.remove(req_id)
                    batch_jobs.pop()  # Remove the last added job
                    batch_request_ids.remove(req_id)
                    waiting_prompts.add(req_id)
                    break

            if batch_jobs:
                # Calculate Average Context Length
                total_context_length = sum(job['context_length'] for job in batch_jobs)
                batch_size = len(batch_jobs)
                average_context_length = total_context_length / batch_size

                # Calculate Total Input of Prompts in Batch
                total_input_of_prompts = sum(job['input_size'] for job in batch_jobs)

                # Calculate Processing Time using the new formula
                processing_time = (
                    (0.0027 * average_context_length + 0.52) * batch_size +
                    44.6 + 0.378 * total_input_of_prompts
                ) / 1000  # Convert milliseconds to seconds

                batch_end_time = current_time + processing_time

                # Update machine state
                machine_busy = True
                batch_in_progress = {
                    'start_time': current_time,
                    'end_time': batch_end_time,
                    'requests': [job['req_id'] for job in batch_jobs],
                    'size': batch_size
                }
                # Uncomment the following line if you want to see the batches being processed
                # print(f"Batch starting at {current_time} with processing time {processing_time}s, batch size {batch_size}")
            else:
                pass

        # Update memory usage
        total_memory_usage = 0
        for req in requests:
            if req.completed:
                continue
            if req.started:
                total_memory_usage += req.input_size + req.tokens_processed
        memory_usage_over_time.append((current_time, total_memory_usage))

        num_task = 0
        for req in requests:
            if req.started and not req.completed:
                num_task += 1
        num_of_task_over_time.append(num_task)

        if total_memory_usage > M:
            print(f"Memory limit exceeded at time {current_time}, usage: {total_memory_usage}")
            break  # Stop the simulation if memory limit is exceeded

    # After simulation, calculate total latency
    total_latency = 0.0
    completed_request_count = 0
    for req in requests:
        if req.completed:
            latency = req.finish_time - req.arrival_time
            total_latency += latency
            completed_request_count += 1

    averaged_latency_list.append(total_latency / completed_request_count)

print(averaged_latency_list)
