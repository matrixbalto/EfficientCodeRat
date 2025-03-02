# BratRats - Code Generation and Evaluation

A web interface for querying a reinforcement learning model (Llama 3.1 8B) with coding problems, generating solutions, and evaluating them using a sandbox evaluator.

## Overview

This project provides an elegant web interface that allows users to:

1. Submit coding problems and test cases
2. Generate code solutions using a reinforcement learning model
3. Run the code using a backend evaluator
4. Display the evaluation results with runtime metrics

The model is trained using Advantage Weighted Regression (AWR) on a base foundation model (Llama 3.1 8B) and is evaluated on the Enamel and Mercury benchmarks.

## Repository Structure

- `frontend/`: React-based web interface
- `web_api.py`: Flask API for handling requests
- `sandbox.py`: Code execution sandbox for evaluating solutions
- `benchmark.py`: Benchmarking utilities for model evaluation
- `prompts/`: Prompt templates for model inference

## Setup Instructions

### Prerequisites

- Python 3.10+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/couplefire/bratrats.git
   cd bratrats
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

5. Start the backend server:
   ```bash
   python web_api.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

3. Start the frontend development server:
   ```bash
   npm start
   # or
   yarn start
   ```

4. The web interface will be available at http://localhost:3000

### Running the Complete Interface

You can use the provided script to start both the backend and frontend servers:

```bash
chmod +x run_web_interface.sh
./run_web_interface.sh
```

## Usage

1. Enter a coding problem in the problem description field
2. Provide test cases in JSON format
3. Submit the problem
4. Generate a solution
5. Evaluate the solution against the test cases
6. View the evaluation results, including runtime metrics

## Test Case Format

Test cases should be provided in the following JSON format:

```json
[
  {
    "input": [1, 2, 3],
    "expected": 6
  },
  {
    "input": [4, 5, 6],
    "expected": 15
  }
]
```

## Development

### Backend Development

The backend is built with Flask and provides the following API endpoints:

- `/api/submit`: Submit a coding problem and test cases
- `/api/generate`: Generate a solution for a submitted problem
- `/api/evaluate`: Evaluate a generated solution against test cases
- `/api/status/<job_id>`: Get the status of a job

### Frontend Development

The frontend is built with React and uses the following components:

- `App.js`: Main application component
- `ProblemInput.js`: Component for entering problem descriptions
- `TestCaseInput.js`: Component for entering test cases
- `CodeEditor.js`: Component for displaying and editing code
- `ResultDisplay.js`: Component for displaying evaluation results

## License

[MIT License](LICENSE)

