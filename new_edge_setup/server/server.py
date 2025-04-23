from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import torch
import logging
import ssl
import multiprocessing
import time
import random
import queue
import threading

from model_initialisation import Model


MODEL_CONFIGS = [
    {'model_type': 'ultralytics_yolo', 'path': '/home/isat/Aditya_Pachpande/ModelInference_Setup/server/models/object_detection.pt', 'device': 'cuda', 'id': 'object_detection'}

]

INPUT_QUEUE_MAX_SIZE = 20
OUTPUT_QUEUE_MAX_SIZE = 50
WORKER_TIMEOUT = 5.0
RESULT_TIMEOUT = 5.0


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s:%(threadName)s] - %(message)s')
log_flask = logging.getLogger('werkzeug')
log_flask.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


if not os.path.exists('models'):
    os.makedirs('models')
    logger.info("Created 'models' directory.")

for config in MODEL_CONFIGS:
    path = config['path']
    if config['model_type'] == 'ultralytics_yolo' and not os.path.exists(path):
        logger.warning(f"Creating dummy file for {path}. Replace with your actual model.")
        try:
            with open(path, 'w') as f:
                f.write("This is a placeholder model file.")
        except IOError as e:
             logger.error(f"Failed to create dummy file {path}: {e}")
    elif config['model_type'] == 'VLM' and not os.path.exists(path):
         logger.warning(f"Creating dummy directory for {path}. Replace with your actual model files.")
         try:
             os.makedirs(path)
         except OSError as e:
             logger.error(f"Failed to create dummy directory {path}: {e}")


app = Flask(__name__)
CORS(app)


def model_worker(model_config: dict, input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue):
    multiprocessing.current_process().name = f"Worker-{model_config.get('id', 'Unknown')}"

    model_id = model_config.get('id', model_config['model_type'])
    pid = os.getpid()
    device = model_config.get('device', 'cpu')
    logging.info(f"Worker starting: {model_id} (PID: {pid}) on device {device}")

    model_wrapper = Model(
        model_type=model_config['model_type'],
        path=model_config['path'],
        device=device
    )
    model = model_wrapper.load_model()

    if model is None:
        logging.error(f"Worker {model_id} (PID: {pid}) failed to load model. Exiting.")
        try:
            output_queue.put((model_id, {"error": "Model loading failed"}, None), timeout=1.0)
        except queue.Full:
            logging.warning(f"Worker {model_id} (PID: {pid}): Output queue full when trying to report load error.")
        return

    logging.info(f"Worker ready: {model_id} (PID: {pid})")

    while True:
        request_id = None
        try:
            request_data = input_queue.get(timeout=WORKER_TIMEOUT)

            if request_data is None:
                logging.info(f"Worker stopping: {model_id} (PID: {pid}) received stop signal.")
                break

            request_id, data_to_process = request_data
            logging.info(f"Worker {model_id} (PID: {pid}) processing request ID: {request_id}")

            start_time = time.time()
            result = model_wrapper.predict(data_to_process)
            end_time = time.time()
            processing_time = end_time - start_time

            if result is not None:
                 logging.info(f"Worker {model_id} (PID: {pid}) finished request ID: {request_id} in {processing_time:.3f}s")
            else:
                 logging.warning(f"Worker {model_id} (PID: {pid}) prediction failed for request ID: {request_id}. Check previous logs.")

            output_item = (model_id, result, request_id)
            output_queue.put(output_item)

        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error in worker {model_id} (PID: {pid}) loop: {e}", exc_info=True)
            try:
                output_queue.put((model_id, {"error": str(e)}, request_id), timeout=1.0)
            except queue.Full:
                 logging.warning(f"Worker {model_id} (PID: {pid}): Output queue full when trying to report processing error.")
            except Exception as eq:
                 logging.error(f"Worker {model_id} (PID: {pid}): Error reporting error to queue: {eq}", exc_info=True)
            continue

    logging.info(f"Worker finished: {model_id} (PID: {pid})")


manager = multiprocessing.Manager()
input_data_queue = manager.Queue(maxsize=INPUT_QUEUE_MAX_SIZE)
results_queue = manager.Queue(maxsize=OUTPUT_QUEUE_MAX_SIZE)
processes = []
request_counter = manager.Value('i', 0)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if not request.is_json:
        logging.warning(f"Rejecting non-JSON request from {request.remote_addr}")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    input_data = data.get('data')

    if input_data is None:
        logging.warning(f"Rejecting request from {request.remote_addr} due to missing 'data' field.")
        return jsonify({"error": "Missing 'data' field in JSON payload"}), 400

    with request_counter.get_lock():
        current_id = request_counter.value
        request_counter.value += 1

    try:
        input_item = (current_id, input_data)
        input_data_queue.put(input_item, block=True, timeout=1.0)
        logging.info(f"Enqueued request ID {current_id} from {request.remote_addr} with data snippet: {str(input_data)[:50]}...")
        return jsonify({"message": "Request accepted", "request_id": current_id}), 202
    except queue.Full:
        logging.error("Input queue is full. Request rejected.")
        return jsonify({"error": "Server busy, input queue full"}), 503
    except Exception as e:
        logging.exception(f"Error enqueuing request ID {current_id}: {e}")
        return jsonify({"error": "Internal server error during enqueue"}), 500


def result_collector(output_q: multiprocessing.Queue):
    threading.current_thread().name = "ResultCollector"
    logging.info("Result collector thread started.")
    processed_requests = set()

    while True:
        try:
            model_id, result, request_id = output_q.get(timeout=RESULT_TIMEOUT)

            if request_id is not None:
                 logging.info(f"Received result for Request ID {request_id} from Model '{model_id}'.")
                 logging.info(f"  > Result Snippet: {str(result)[:100]}...")

            else:
                logging.warning(f"Received untagged result/message from Model '{model_id}': {result}")

        except queue.Empty:
            if not any(p.is_alive() for p in processes):
                try:
                    model_id, result, request_id = output_q.get_nowait()
                    logging.info(f"Received final result for Request ID {request_id} from Model '{model_id}' after worker exit.")
                    logging.info(f"  > Final Result Snippet: {str(result)[:100]}...")
                except queue.Empty:
                    logging.info("Result Collector: All worker processes exited and output queue is empty. Stopping collector.")
                    break
                except Exception as e:
                    logging.error(f"Error processing final item in queue: {e}", exc_info=True)
                    break
            continue
        except Exception as e:
            logging.error(f"Error in result collector: {e}", exc_info=True)
            time.sleep(1)

    logging.info("Result collector thread finished.")


if __name__ == "__main__":
    multiprocessing.current_process().name = "MainProcess"
    logging.info("Starting server application...")

    logging.info(f"Initializing {len(MODEL_CONFIGS)} model workers...")
    for config in MODEL_CONFIGS:
        try:
            process = multiprocessing.Process(
                target=model_worker,
                args=(config, input_data_queue, results_queue),
                daemon=True,
                name=f"Worker-{config.get('id', 'Unknown')}"
            )
            processes.append(process)
            process.start()
            logging.info(f"Started worker process for {config.get('id', config['model_type'])} with PID {process.pid}")
        except Exception as e:
            logging.exception(f"Error starting worker for {config.get('id', config['model_type'])}: {e}")

    logging.info("Model workers initialized.")

    collector_thread = threading.Thread(target=result_collector, args=(results_queue,), daemon=True, name="ResultCollectorThread")
    collector_thread.start()

    logging.info("Starting Flask server on http://0.0.0.0:8080")
    try:
        app.run(host='0.0.0.0', port=8080, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
         logging.info("Flask server received KeyboardInterrupt.")
    except Exception as e:
         logging.exception("Flask server stopped due to an error.")

    logging.info("Initiating shutdown sequence...")

    logging.info("Sending stop signals (None) to workers via input queue...")
    for i in range(len(processes)):
        try:
            input_data_queue.put(None, timeout=1.0)
        except queue.Full:
            logging.warning(f"Input queue full while sending stop signal {i+1}/{len(processes)}. Some workers might not receive it.")
        except Exception as e:
            logging.error(f"Error sending stop signal: {e}", exc_info=True)

    logging.info("Waiting for worker processes to join...")
    active_processes = True
    join_timeout = 5.0
    shutdown_wait_limit = time.time() + (len(processes) * join_timeout + 10)

    while active_processes and time.time() < shutdown_wait_limit:
        active_processes = False
        finished_processes = 0
        for process in processes:
            if process.is_alive():
                active_processes = True
                process.join(timeout=0.1)
            else:
                 finished_processes += 1
        if active_processes:
            logging.info(f"Waiting for {len(processes) - finished_processes} worker(s) to join...")
            time.sleep(0.5)

    for process in processes:
        if process.is_alive():
            logging.warning(f"Process {process.name} (PID: {process.pid}) did not join cleanly, terminating.")
            try:
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    logging.error(f"Process {process.name} (PID: {process.pid}) failed to terminate.")
            except Exception as e:
                logging.error(f"Error terminating process {process.name} (PID: {process.pid}): {e}", exc_info=True)

    logging.info("Waiting for result collector thread to finish...")
    collector_thread.join(timeout=5.0)
    if collector_thread.is_alive():
        logging.warning("Result collector thread did not finish cleanly.")

    logging.info("Shutdown complete. Main process finished.")