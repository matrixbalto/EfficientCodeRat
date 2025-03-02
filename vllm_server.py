import os
import subprocess
import json
import time
import signal
from threading import Thread, Event, Lock
from contextlib import redirect_stdout
from pathlib import Path

from dotenv import load_dotenv
import argparse
from flask import Response, request, Flask
from flask_cors import CORS
from openai import OpenAI
import torch


load_dotenv()
VLLM_SERVER_HOST = os.getenv('VLLM_SERVER_HOST')
VLLM_SERVER_PORT = os.getenv('VLLM_SERVER_PORT')
VLLM_MODEL_PORT = os.getenv('VLLM_MODEL_PORT')
LOG_FILE_PATH = None

app = Flask(__name__)
CORS(app)


base_model = None
lora_dir = None
cur_checkpoint = None
cur_process = None
cur_checkpoint_list = []
openai_client = None
quantization = None
shutdown_init_mutex = Lock()
is_shutting_down = False

print_stop_event = Event()


def line_dump(obj: any, debug = False) -> str:
    if debug:
        with open(LOG_FILE_PATH, 'a') as f:
            with redirect_stdout(f):
                print(obj, flush=True)
    return json.dumps(obj)+'\n'


def print_process_output(process):
    for line in iter(process.stdout.readline, b''):
        if print_stop_event.is_set():
            break
        with open(LOG_FILE_PATH, 'a') as f:
            with redirect_stdout(f):
                print(line.decode(), flush=True)


@app.route('/shutdown', methods=['POST'])
def shutdown():
    global is_shutting_down

    if is_shutting_down:
        return Response(
            mimetype='application/json',
            status=200, 
            response=line_dump("server shutdown already initiated")
        )

    def shutdown_server():
        global cur_process, openai_client, shutdown_init_mutex, is_shutting_down

        cur_time = time.time()
        with open(LOG_FILE_PATH, 'a') as f:
            with redirect_stdout(f):
                print('shutting down the server..', flush=True)

        shutdown_init_mutex.acquire()
        is_shutting_down = True

        cur_time = time.time()
        with open(LOG_FILE_PATH, 'a') as f:
            with redirect_stdout(f):
                print(f'mutex acquired in {time.time() - cur_time:.2f} seconds', flush=True)

        if cur_process is not None:
            print_stop_event.set()
            os.killpg(os.getpgid(cur_process.pid), signal.SIGKILL)
            cur_process.wait()

        cur_process = None
        openai_client = None

        is_shutting_down = False
        shutdown_init_mutex.release()

        shutdown_time = time.time() - cur_time
        with open(LOG_FILE_PATH, 'a') as f:
            with redirect_stdout(f):
                print(f'server shutdown in {shutdown_time:.2f} seconds', flush=True)

    # Start the shutdown process in a separate thread
    Thread(target=shutdown_server).start()

    return Response(
        mimetype='application/json',
        status=200, 
        response=line_dump("server shutdown initiated")
    )


@app.route('/set_base_model', methods=['POST'])
def set_base_model():
    global base_model, cur_process

    if request.json['base_model'] != base_model: 
        if cur_process is not None:
            shutdown()

        base_model = request.json['base_model']  

    return Response(
        mimetype='application/json',
        status=200, 
        response=line_dump("base_model set")
    )


@app.route('/set_lora_dir', methods=['POST'])
def set_lora_dir():
    global lora_dir

    lora_dir = request.json['lora_dir']

    return Response(
        mimetype='application/json',
        status=200, 
        response=line_dump("lora_dir set")
    )


@app.route('/set_checkpoint_list', methods=['POST'])
def set_checkpoint_list():
    global cur_checkpoint_list, cur_process

    new_checkpoint_list = request.json['checkpoint_list']

    if any([checkpoint not in cur_checkpoint_list for checkpoint in new_checkpoint_list]):
        if cur_process is not None:
            shutdown()
        cur_checkpoint_list = new_checkpoint_list

    return Response(
        mimetype='application/json',
        status=200, 
        response=line_dump("checkpoint_list set")
    )


@app.route('/set_quantization', methods=['POST'])
def set_quantization():
    global quantization

    quantization = request.json['quantization']

    return Response(
        mimetype='application/json',
        status=200, 
        response=line_dump("quantization set")
    )


def init_gen(): 
    global base_model, lora_dir, cur_process, openai_client, quantization, shutdown_init_mutex, cur_checkpoint_list, is_shutting_down

    if base_model is None:
        raise ValueError('base_model is not set')
    
    if lora_dir is None:
        raise ValueError('lora_dir is not set')
    
    if (not is_shutting_down) and (cur_process is not None):
        return
    
    shutdown_init_mutex.acquire()

    if (not is_shutting_down) and (cur_process is not None):
        return

    checkpoints = [f'checkpoint-{checkpoint}' for checkpoint in cur_checkpoint_list]
    lora_modules = [f'{checkpoint}={os.path.join(lora_dir, checkpoint)}' for checkpoint in checkpoints]

    yield 'starting the new server..\n'
    args_list = [
        'vllm', 'serve', base_model, 
        '--enable-lora', 
        '--max-lora-rank', '64',
        '--lora-modules', *lora_modules,
        '--port', str(int(VLLM_MODEL_PORT) + args.port_shift), 
        '--disable-log-requests', 
        '--enable-prefix-caching'
    ]

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1: 
        args_list.append('--tensor_parallel_size')
        args_list.append(str(num_gpus))

    if quantization is not None:
        args_list.append('--quantization')
        args_list.append(quantization)
    cur_process = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
    yield 'new server started\n'

    print_stop_event.clear()
    Thread(target=print_process_output, args=(cur_process,), daemon=True).start()

    openai_client = OpenAI(
        base_url=f'http://0.0.0.0:{int(VLLM_MODEL_PORT) + args.port_shift}/v1',
    )

    # we wait until the client is ready
    while True:
        time.sleep(1)
        try:
            openai_client.completions.create(
                model=base_model, 
                prompt='Hello, world!'
            )
            break
        except Exception as e:
            yield f'client is not ready yet: {e}\n'
    yield 'client is ready\n'

    shutdown_init_mutex.release()


@app.route('/init', methods=['POST'])
def init():
    return Response(
        mimetype='application/json',
        status=200, 
        response=map(line_dump, init_gen())
    )


@app.route('/get_checkpoints', methods=['POST'])
def get_checkpoints():
    global cur_checkpoint_list

    return Response(
        mimetype='application/json',
        status=200,
        response=line_dump({
            'checkpoints': cur_checkpoint_list
        })
    )


@app.route('/set_checkpoint', methods=['POST'])
def set_checkpoint():
    global cur_checkpoint, cur_checkpoint_list

    cur_checkpoint = request.json['checkpoint']
    assert (cur_checkpoint in cur_checkpoint_list)

    return Response(
        mimetype='application/json',
        status=200, 
        response=line_dump("checkpoint set")
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_list', nargs='+', default=None)
    parser.add_argument('--port_shift', type=int, default=0)
    parser.add_argument('--base_model', type=str, default=None)
    parser.add_argument('--lora_dir', type=str, default=None)
    parser.add_argument('--checkpoint_list', nargs='+', default=None)
    parser.add_argument('--default_checkpoint', type=str, default=None)
    parser.add_argument('--quantization', type=str, default=None)

    args = parser.parse_args()

    if args.device_list is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.device_list)

    cur_time = time.time()
    home = str(Path.home())
    LOG_FILE_PATH = f'{home}/logs/vllm_server_{cur_time}.log'
    if not os.path.exists(f'{home}/logs'):
        os.makedirs(f'{home}/logs')

    if args.base_model is not None:
        base_model = args.base_model

    if args.lora_dir is not None:
        lora_dir = args.lora_dir

    if args.checkpoint_list is not None:
        cur_checkpoint_list = args.checkpoint_list

    if args.default_checkpoint is not None:
        cur_checkpoint = args.default_checkpoint

    if args.quantization is not None:
        quantization = args.quantization

    try:
        res = init_gen()
        for line in res:
            print(line, end='', flush=True)
    except Exception as e:
        print(f'error: {e}')

    app.run(host='0.0.0.0', port=int(VLLM_SERVER_PORT) + args.port_shift)