import pandas as pd
import numpy as np
import heapq
import random

np.random.seed(42)  # For reproducibility

# Parameters
M = 16492  # Memory capacity
alpha = 0.1  # Parameter for memory check (0 < alpha < 1)
beta = 0.1
averaged_latency_list = []

for i in range(10):
    num_rows = int(1000 * (i + 1))
    df = dfmain.head(num_rows)
    # Initialize variables
    current_time = 0.0
    event_queue = []  # List of events: (time, event_type, data)

    # Create a list to hold the requests
    requests = []

    # Create a class to represent each request
    class Request:
        def __init__(self, request_id, arrival_time, input_size, output_size):
            self.request_id = request_id
            self.arrival_time = arrival_time
            self.input_size = int(input_size)
            self.output_size = int(output_size)
            self.tokens_processed = 0  # Number of tokens processed (including prompt)
            self.started = False  # Whether the prompt has been started processing
            self.completed = False  # Whether the request has been fully processed
            self.finish_time = None  # Time when the last token was processed
            self.start_time = None  # Time when the prompt was first processed
            self.remaining_tokens = self.output_size + 1  # Include prompt processing
            self.context_length = 0  # Added: Initialize context length

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
    tokens_ready = set()  # Requests whose next token is ready to be processed

    # Initialize latency tracking
    total_latency = 0.0

    # Initialize memory usage tracking
    memory_usage_over_time = []
    memory_resets = 0  # Count the number of memory resets

    # Initialize machine state
    machine_busy = False
    batch_end_time = 0.0
    batch_in_progress = None

    heapq.heapify(event_queue)  # Convert event_queue to a heap

    # Main simulation loop
    while True:
        # Check if simulation should end
        if current_time >= time_limit:
            break  # Exit the simulation if time limit is reached

        # Determine the next event
        next_event_time = None
        if machine_busy:
            next_event_time = batch_end_time
        if event_queue:
            if next_event_time is None or event_queue[0][0] < next_event_time:
                next_event_time = event_queue[0][0]

        if next_event_time is None:
            break  # No more events to process

        # Advance time to the next event, but not beyond the time limit
        current_time = min(next_event_time, time_limit)

        # Process events at the current time
        if machine_busy and current_time == batch_end_time:
            # Process batch completion
            batch = batch_in_progress
            # Update requests
            for req_id in batch['requests']:
                req = requests[req_id]
                if not req.started:
                    # This was a prompt
                    req.started = True
                    req.tokens_processed += 1
                    req.start_time = current_time
                    tokens_ready.add(req_id)
                    running_requests.add(req_id)
                    req.remaining_tokens -= 1
                    req.context_length = 1  # Added: Next token will have context length 1
                else:
                    # This was a token
                    req.tokens_processed += 1
                    req.remaining_tokens -= 1
                    req.context_length += 1  # Added: Increment context length
                    if req.remaining_tokens > 0:
                        tokens_ready.add(req_id)
                    else:
                        # Request is completed
                        req.completed = True
                        req.finish_time = current_time
                        running_requests.discard(req_id)
                        tokens_ready.discard(req_id)

            # Clear the batch
            machine_busy = False
            batch_in_progress = None

            # Update memory usage
            total_memory_usage = sum(
                requests[req_id].input_size + requests[req_id].tokens_processed
                for req_id in running_requests
            )
            memory_usage_over_time.append((current_time, total_memory_usage))

            # Check for memory overflow and perform iterative partial resets
            while total_memory_usage > M and current_time < time_limit:
                # Perform partial memory reset based on beta
                memory_resets += 1
                requests_reset = []
                for req_id in list(running_requests):
                    if random.random() < beta:
                        req = requests[req_id]
                        req.tokens_processed = 0
                        req.started = False
                        req.remaining_tokens = req.output_size + 1
                        req.start_time = None
                        req.context_length = 0  # Added: Reset context length
                        # Move back to waiting_prompts
                        waiting_prompts.add(req_id)
                        running_requests.discard(req_id)
                        tokens_ready.discard(req_id)
                        requests_reset.append(req_id)
                # Update total memory usage after partial reset
                total_memory_usage = sum(
                    requests[req_id].input_size + requests[req_id].tokens_processed
                    for req_id in running_requests
                )
                memory_usage_over_time.append((current_time, total_memory_usage))
                print(f"Partial memory reset occurred at time {current_time}, reset {len(requests_reset)} requests")
                if total_memory_usage <= M:
                    break  # Memory usage is acceptable
                else:
                    # Advance time by 1 unit during which no processing occurs
                    next_time = current_time + 1
                    # Handle arrivals that occur during the waiting period
                    while event_queue and event_queue[0][0] <= next_time and event_queue[0][0] <= time_limit:
                        arrival_time, event_type, data = heapq.heappop(event_queue)
                        current_time = arrival_time
                        if event_type == 'arrival':
                            req_id = data
                            waiting_prompts.add(req_id)
                        # Update memory usage (unchanged since no processing)
                        memory_usage_over_time.append((current_time, total_memory_usage))
                    current_time = min(next_time, time_limit)
                    if current_time >= time_limit:
                        break  # Exit if time limit reached

        elif event_queue and current_time == event_queue[0][0]:
            # Process arrival event
            _, event_type, data = heapq.heappop(event_queue)
            if event_type == 'arrival':
                req_id = data
                waiting_prompts.add(req_id)
            else:
                pass

            # Update memory usage
            total_memory_usage = sum(
                requests[req_id].input_size + requests[req_id].tokens_processed
                for req_id in running_requests
            )
            memory_usage_over_time.append((current_time, total_memory_usage))

            # Check for memory overflow and perform iterative partial resets
            while total_memory_usage > M and current_time < time_limit:
                # Perform partial memory reset based on beta
                memory_resets += 1
                requests_reset = []
                for req_id in list(running_requests):
                    if random.random() < beta:
                        req = requests[req_id]
                        req.tokens_processed = 0
                        req.started = False
                        req.remaining_tokens = req.output_size + 1
                        req.start_time = None
                        req.context_length = 0  # Added: Reset context length
                        # Move back to waiting_prompts
                        waiting_prompts.add(req_id)
                        running_requests.discard(req_id)
                        tokens_ready.discard(req_id)
                        requests_reset.append(req_id)
                # Update total memory usage after partial reset
                total_memory_usage = sum(
                    requests[req_id].input_size + requests[req_id].tokens_processed
                    for req_id in running_requests
                )
                memory_usage_over_time.append((current_time, total_memory_usage))
                print(f"Partial memory reset occurred at time {current_time}, reset {len(requests_reset)} requests")
                if total_memory_usage <= M:
                    break  # Memory usage is acceptable
                else:
                    # Advance time by 1 unit during which no processing occurs
                    next_time = current_time + 1
                    # Handle arrivals that occur during the waiting period
                    while event_queue and event_queue[0][0] <= next_time and event_queue[0][0] <= time_limit:
                        arrival_time, event_type, data = heapq.heappop(event_queue)
                        current_time = arrival_time
                        if event_type == 'arrival':
                            req_id = data
                            waiting_prompts.add(req_id)
                        # Update memory usage (unchanged since no processing)
                        memory_usage_over_time.append((current_time, total_memory_usage))
                    current_time = min(next_time, time_limit)
                    if current_time >= time_limit:
                        break  # Exit if time limit reached

        else:
            # No events to process at current time
            pass

        # Check if machine is idle and can start a new batch
        if not machine_busy and current_time < time_limit:
            # Form a batch according to the algorithm
            batch_jobs = []  # Modified: Use batch_jobs instead of batch_tokens
            batch_request_ids = set()

            # Step 1: Include tokens ready to be processed
            for req_id in list(tokens_ready):
                if req_id not in batch_request_ids:
                    req = requests[req_id]
                    batch_jobs.append({
                        'req_id': req_id,
                        'job_type': 'token',
                        'context_length': req.context_length,
                        'input_size': 0  # Tokens do not contribute to total input size
                    })
                    batch_request_ids.add(req_id)
                    tokens_ready.discard(req_id)
                    if len(batch_jobs) >= B:
                        break  # Batch size limit reached

            # Check existing memory usage
            existing_memory_usage = sum(
                requests[req_id].input_size + requests[req_id].tokens_processed
                for req_id in running_requests
            )
            if existing_memory_usage > M * (1 - alpha):
                # Do not add new prompts
                pass
            else:
                # Step 2: Add new prompts from waiting_prompts
                waiting_prompts_list = sorted(waiting_prompts, key=lambda x: requests[x].arrival_time)
                for req_id in waiting_prompts_list:
                    if req_id in batch_request_ids:
                        continue
                    if len(batch_jobs) >= B:
                        break  # Batch size limit reached
                    req = requests[req_id]
                    batch_jobs.append({
                        'req_id': req_id,
                        'job_type': 'prompt',
                        'context_length': 0,  # Prompts have context length 0
                        'input_size': req.input_size
                    })
                    batch_request_ids.add(req_id)
                    waiting_prompts.discard(req_id)

            if batch_jobs:
                # Calculate batch size
                batch_size = len(batch_jobs)
                # Calculate total context length
                total_context_length = sum(job['context_length'] for job in batch_jobs)
                average_context_length = total_context_length / batch_size if batch_size > 0 else 0

                # Calculate total input of prompts
                total_input_of_prompts = sum(job['input_size'] for job in batch_jobs)

                # Calculate processing time using the new formula
                processing_time = (
                    (0.0027 * average_context_length + 0.52) * batch_size +
                    44.6 + 0.378 * total_input_of_prompts
                ) / 1000  # Convert milliseconds to seconds

                batch_end_time = current_time + processing_time
                # Ensure batch_end_time does not exceed time_limit
                if batch_end_time > time_limit:
                    batch_end_time = time_limit
                    processing_time = batch_end_time - current_time
                # Update machine state
                machine_busy = True
                batch_in_progress = {
                    'start_time': current_time,
                    'end_time': batch_end_time,
                    'requests': [job['req_id'] for job in batch_jobs],
                    'size': batch_size
                }
                # No need to add to event_queue since we are managing time
            else:
                pass

    # After simulation, calculate total latency
    total_latency = 0.0
    completed_request_count = 0
    for req in requests:
        if req.completed:
            latency = req.finish_time - req.arrival_time
            total_latency += latency
            completed_request_count += 1
    print(total_latency / completed_request_count)
    averaged_latency_list.append(total_latency / completed_request_count)

print(averaged_latency_list)
